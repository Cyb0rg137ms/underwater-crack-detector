from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import io
import tempfile
import uuid
import json
from datetime import datetime
from skimage.morphology import remove_small_objects

# Initialize Flask app
app = Flask(__name__, 
    static_folder='static',
    template_folder='build')
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
MODELS_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global model variables
crack_model = None

# Model Architecture
class HLFSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2]):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=r, dilation=r, bias=False)
            for r in dilation_rates
        ])
        self.bn = nn.BatchNorm2d(out_channels * (len(dilation_rates) + 1))
        self.fusion = nn.Conv2d(out_channels * (len(dilation_rates) + 1), out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.contrast_guide = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.depthwise(x)
        base_feat = self.pointwise(x)
        dilated_feats = [base_feat]
        for conv in self.dilated_convs:
            dilated_feats.append(conv(base_feat))
        fused = torch.cat(dilated_feats, dim=1)
        fused = self.bn(fused)
        fused = self.fusion(fused)
        guide = self.contrast_guide(fused)
        output = self.relu(fused * guide)
        return output

class EnhancedAttentionGate(nn.Module):
    def __init__(self, in_channels, gate_channels, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or in_channels // 2
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=False)
        spatial_attn = self.psi(F.relu(g1 + x1, inplace=True))
        return x * spatial_attn

class CrackDetectionNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.encoder1 = HLFSBlock(32, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = HLFSBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = HLFSBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bridge = nn.Sequential(
            HLFSBlock(256, 512),
            nn.Dropout2d(0.4)
        )
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att3 = EnhancedAttentionGate(256, 512)
        self.decoder3 = HLFSBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att2 = EnhancedAttentionGate(128, 256)
        self.decoder2 = HLFSBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att1 = EnhancedAttentionGate(64, 128)
        self.decoder1 = HLFSBlock(128, 64)
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

        # Ensure all parameters are contiguous
        for param in self.parameters():
            param.data = param.data.contiguous()

    def forward(self, x):
        x1 = self.input_block(x)
        e1 = self.encoder1(x1)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        bridge = self.bridge(p3)
        u3 = self.upconv3(bridge)
        a3 = self.att3(e3, bridge)
        d3 = self.decoder3(torch.cat([u3, a3], dim=1))
        u2 = self.upconv2(d3)
        a2 = self.att2(e2, d3)
        d2 = self.decoder2(torch.cat([u2, a2], dim=1))
        u1 = self.upconv1(d2)
        a1 = self.att1(e1, d2)
        d1 = self.decoder1(torch.cat([u1, a1], dim=1))
        return self.output(d1)

# Load models
def load_models():
    global crack_model
    
    # Load crack detection model
    model_path = os.path.join(MODELS_FOLDER, 'crack_detector_weights.pth')
    if os.path.exists(model_path):
        try:
            print(f"Loading model from {model_path}")
            
            # Initialize the model with the correct architecture
            crack_model = CrackDetectionNetwork()
            
            # Load the state dict
            checkpoint = torch.load(model_path, map_location=device)
            if "model_state_dict" in checkpoint:
                # If the file is a checkpoint dict with 'model_state_dict' key
                crack_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # If the file is just the state dict itself
                crack_model.load_state_dict(checkpoint)
                
            crack_model.to(device)
            crack_model.eval()
            print("Crack detection model loaded successfully")
        except Exception as e:
            print(f"Error loading crack model: {e}")
            crack_model = None
    else:
        print(f"Model file not found at {model_path}")
        crack_model = None

# Preprocessing 
def preprocess_image(image):
    """Preprocess image for model inference"""
    # Convert to PIL image if it's a NumPy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Apply CLAHE enhancement
    img_np = np.array(image)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    enhanced_image = Image.fromarray(img_np)
    
    # Apply transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(enhanced_image).unsqueeze(0), enhanced_image

# Post-processing
def advanced_post_processing(mask, min_size=30, threshold=0.5):
    """Apply post-processing to the model output mask"""
    binary_mask = (mask > threshold).astype(np.uint8)
    cleaned = remove_small_objects(binary_mask.astype(bool), min_size=min_size)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed

# Analyze crack severity
def analyze_crack_severity(mask):
    """Analyze crack severity from mask"""
    crack_area = np.sum(mask)
    total_area = mask.size
    percentage = (crack_area / total_area) * 100
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "No cracks", 0.0, [], 0
    
    lengths = [cv2.arcLength(c, True) for c in contours if cv2.contourArea(c) > 5]
    if not lengths:
        return "No significant cracks", 0.0, [], 0
    
    max_length = max(lengths)
    total_length = sum(lengths)
    num_significant = len(lengths)
    severity_score = (percentage * 0.4) + (max_length / 100) * 0.6
    
    if severity_score > 2.0 or max_length > 70:
        return "Critical cracks", percentage, contours, 3
    elif severity_score > 1.0 or max_length > 40:
        return "Moderate cracks", percentage, contours, 2
    elif severity_score > 0.3:
        return "Minor cracks", percentage, contours, 1
    else:
        return "Negligible cracks", percentage, contours, 0

# Process image and detect cracks
def process_image(image_path, threshold=0.5, min_size=30):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return None, "Failed to load image"
    
    # Check if model is loaded
    if crack_model is None:
        return None, "Model not loaded"
    
    # Preprocess image
    image_tensor, enhanced_image = preprocess_image(image)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        logits = crack_model(image_tensor)
        probabilities = torch.sigmoid(logits)
        mask = probabilities.squeeze().cpu().numpy()
    
    # Post-process
    processed_mask = advanced_post_processing(mask, min_size, threshold)
    
    # Analyze crack severity
    result_text, percentage, contours, severity = analyze_crack_severity(processed_mask)
    
    # Prepare visualization
    original = np.array(enhanced_image.resize((256, 256)))
    result_image = original.copy()
    
    # Color-code based on severity
    color_map = {0: (0, 255, 0), 1: (255, 255, 0), 2: (0, 165, 255), 3: (0, 0, 255)}
    color = color_map[severity]
    
    # Draw contours
    if contours:
        cv2.drawContours(result_image, contours, -1, color, 2)
    
    # Create overlay with crack areas
    overlay = result_image.copy()
    for c in range(3):
        overlay[:,:,c] = np.where(processed_mask > 0, color[c], overlay[:,:,c])
    
    # Blend overlay with original
    alpha = 0.3
    result_image = cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0)
    
    # Generate unique result filename
    result_filename = f"result_{uuid.uuid4()}.jpg"
    result_path = os.path.join(RESULTS_FOLDER, result_filename)
    
    # Save result image
    cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    
    # Prepare results for frontend
    results = {
        'id': 1,
        'width': round(percentage * 0.1, 2),  # Approximate width based on percentage
        'height': round(percentage * 0.05, 2),  # Approximate height
        'position': {
            'x': 0,
            'y': 0,
            'w': 256,
            'h': 256
        },
        'confidence': float(percentage / 100),
        'severity': ["negligible", "minor", "moderate", "critical"][severity]
    }
    
    return f"/results/{result_filename}", [results] if severity > 0 else []

# Process video
def process_video(video_path, threshold=0.5, min_size=30):
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Failed to open video"
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Generate unique result filename
    result_filename = f"result_{uuid.uuid4()}.mp4"
    result_path = os.path.join(RESULTS_FOLDER, result_filename)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (256, 256))
    
    # Process frames
    frame_count = 0
    max_frames = 300  # Limit to prevent very long processing
    all_results = []
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5 frames to speed up
        if frame_count % 5 == 0:
            # Preprocess frame
            image_tensor, enhanced_image = preprocess_image(frame)
            image_tensor = image_tensor.to(device)
            
            # Run inference
            with torch.no_grad():
                logits = crack_model(image_tensor)
                probabilities = torch.sigmoid(logits)
                mask = probabilities.squeeze().cpu().numpy()
            
            # Post-process
            processed_mask = advanced_post_processing(mask, min_size, threshold)
            
            # Analyze crack severity
            result_text, percentage, contours, severity = analyze_crack_severity(processed_mask)
            
            # Convert enhanced image to numpy for visualization
            original = np.array(enhanced_image)
            result_image = original.copy()
            
            # Color-code based on severity
            color_map = {0: (0, 255, 0), 1: (255, 255, 0), 2: (0, 165, 255), 3: (0, 0, 255)}
            color = color_map[severity]
            
            # Draw contours
            if contours:
                cv2.drawContours(result_image, contours, -1, color, 2)
            
            # Create overlay with crack areas
            overlay = result_image.copy()
            for c in range(3):
                overlay[:,:,c] = np.where(processed_mask > 0, color[c], overlay[:,:,c])
            
            # Blend overlay with original
            alpha = 0.3
            result_image = cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0)
            
            # Add result to frame
            if severity > 0:
                result = {
                    'id': len(all_results) + 1,
                    'width': round(percentage * 0.1, 2),
                    'height': round(percentage * 0.05, 2),
                    'position': {
                        'x': 0,
                        'y': 0,
                        'w': 256,
                        'h': 256
                    },
                    'confidence': float(percentage / 100),
                    'severity': ["negligible", "minor", "moderate", "critical"][severity]
                }
                all_results.append(result)
            
            # Write frame to output
            out.write(cv2.resize(result_image, (256, 256)))
        else:
            # For skipped frames, just resize and write
            resized_frame = cv2.resize(frame, (256, 256))
            out.write(resized_frame)
        
        frame_count += 1
    
    # Release resources
    cap.release()
    out.release()
    
    return f"/results/{result_filename}", all_results

# API Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Get parameters
    threshold = float(request.form.get('threshold', 0.5))
    min_size = int(request.form.get('min_size', 30))
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + '_' + file.filename)
    file.save(file_path)
    
    try:
        # Process file based on type
        if file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            result_path, detections = process_image(file_path, threshold, min_size)
        elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            result_path, detections = process_video(file_path, threshold, min_size)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({
            'success': True,
            'result': result_path,
            'detections': detections,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera', methods=['POST'])
def process_camera():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No captured image'}), 400
    
    # Get parameters
    threshold = float(request.form.get('threshold', 0.5))
    min_size = int(request.form.get('min_size', 30))
    
    # Save captured image
    file_path = os.path.join(UPLOAD_FOLDER, f"camera_{uuid.uuid4()}.jpg")
    file.save(file_path)
    
    try:
        # Process image
        result_path, detections = process_image(file_path, threshold, min_size)
        
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({
            'success': True,
            'result': result_path,
            'detections': detections,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_model', methods=['POST'])
def update_model():
    if 'model' not in request.files:
        return jsonify({'error': 'No model file part'}), 400
    
    model_file = request.files['model']
    if model_file.filename == '':
        return jsonify({'error': 'No selected model file'}), 400
    
    if not model_file.filename.endswith('.pth'):
        return jsonify({'error': 'Invalid model file format'}), 400
    
    # Save model file
    model_path = os.path.join(MODELS_FOLDER, 'crack_detector_weights.pth')
    model_file.save(model_path)
    
    # Verify file was saved
    file_size = os.path.getsize(model_path)
    print(f"Model file saved at {model_path} with size {file_size} bytes")
    
    try:
        # Reload models
        load_models()
        return jsonify({'message': f'Model updated successfully. File size: {file_size} bytes'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<path:filename>')
def result_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

# Initialize models on startup
with app.app_context():
    load_models()

# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)