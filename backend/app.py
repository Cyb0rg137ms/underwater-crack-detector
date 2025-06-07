from flask import Flask, request, jsonify, send_from_directory, render_template
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
import logging
from threading import Thread
from time import sleep

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get port from environment variable (for Render.com)
port = int(os.environ.get("PORT", 5000))

# Initialize Flask app
app = Flask(__name__, 
    static_folder='build/static',  
    template_folder='build')       
CORS(app, resources={r"/api/*": {"origins": "*"}})  

# Configure folders
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
MODELS_FOLDER = 'models'
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, MODELS_FOLDER, 'static']:
    os.makedirs(folder, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Global model variables
crack_model = None
task_results = {}

# Model Architecture
class HLFSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2]):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dilated_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=r, dilation=r, bias=False) for r in dilation_rates])
        self.bn = nn.BatchNorm2d(out_channels * (len(dilation_rates) + 1))
        self.fusion = nn.Conv2d(out_channels * (len(dilation_rates) + 1), out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.contrast_guide = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.Sigmoid())

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
        return self.relu(fused * guide)

class EnhancedAttentionGate(nn.Module):
    def __init__(self, in_channels, gate_channels, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or in_channels // 2
        self.W_g = nn.Sequential(nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False), nn.BatchNorm2d(inter_channels))
        self.W_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False), nn.BatchNorm2d(inter_channels))
        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())

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
        self.input_block = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.encoder1 = HLFSBlock(32, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = HLFSBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = HLFSBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bridge = nn.Sequential(HLFSBlock(256, 512), nn.Dropout2d(0.4))
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
    model_path = os.path.join(MODELS_FOLDER, 'crack_detector_weights.pth')
    if os.path.exists(model_path):
        try:
            logger.info(f"Loading model from {model_path}")
            crack_model = CrackDetectionNetwork()
            checkpoint = torch.load(model_path, map_location=device)
            if "model_state_dict" in checkpoint:
                crack_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                crack_model.load_state_dict(checkpoint)
            crack_model.to(device)
            crack_model.eval()
            logger.info("Crack detection model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading crack model: {e}")
            crack_model = None
    else:
        logger.warning(f"Model file not found at {model_path}")
        crack_model = None

# Initialize function
def initialize():
    if not os.path.exists('build'):
        logger.warning("Build folder not found! Frontend assets might be missing.")
    load_models()

# Preprocessing
def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_np = np.array(image)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    enhanced_image = Image.fromarray(img_np)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform(enhanced_image).unsqueeze(0), enhanced_image

# Post-processing
def advanced_post_processing(mask, min_size=30, threshold=0.5):
    binary_mask = (mask > threshold).astype(np.uint8)
    cleaned = remove_small_objects(binary_mask.astype(bool), min_size=min_size)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed

# Analyze crack severity
def analyze_crack_severity(mask, image_size=(256, 256)):
    crack_area = np.sum(mask) * (image_size[0] * image_size[1] / 1000000)
    total_area = mask.size
    percentage = (crack_area / total_area) * 100
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "No cracks", 0.0, [], 0, 0.0
    lengths = [cv2.arcLength(c, True) for c in contours if cv2.contourArea(c) > 5]
    if not lengths:
        return "No significant cracks", 0.0, [], 0, 0.0
    max_length = max(lengths)
    severity_score = (percentage * 0.4) + (max_length / 100) * 0.6
    if severity_score > 2.0 or max_length > 70:
        return "Critical cracks", percentage, contours, 3, crack_area
    elif severity_score > 1.0 or max_length > 40:
        return "Moderate cracks", percentage, contours, 2, crack_area
    elif severity_score > 0.3:
        return "Minor cracks", percentage, contours, 1, crack_area
    else:
        return "Negligible cracks", percentage, contours, 0, crack_area

# Process image asynchronously
def process_image_async(image_path, threshold=0.5, min_size=30):
    logger.info(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None, "Failed to load image"
    if crack_model is None:
        logger.error("Crack detection model not loaded")
        return None, "Model not loaded"
    image_tensor, enhanced_image = preprocess_image(image)
    image_tensor = image_tensor.to(device)
    sleep(1)  # Simulate initial processing
    with torch.no_grad():
        logits = crack_model(image_tensor)
        probabilities = torch.sigmoid(logits)
        mask = probabilities.squeeze().cpu().numpy()
    processed_mask = advanced_post_processing(mask, min_size, threshold)
    result_text, percentage, contours, severity, crack_area = analyze_crack_severity(processed_mask)
    original = np.array(enhanced_image.resize((256, 256)))
    result_image = original.copy()
    color_map = {0: (0, 255, 0), 1: (255, 255, 0), 2: (0, 165, 255), 3: (0, 0, 255)}
    color = color_map[severity]
    if contours:
        cv2.drawContours(result_image, contours, -1, color, 2)
    overlay = result_image.copy()
    for c in range(3):
        overlay[:,:,c] = np.where(processed_mask > 0, color[c], overlay[:,:,c])
    result_image = cv2.addWeighted(overlay, 0.3, result_image, 0.7, 0)
    result_filename = f"result_{uuid.uuid4()}.jpg"
    result_path = os.path.join(RESULTS_FOLDER, result_filename)
    cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    results = {
        'id': 1,
        'crack_area_cm2': round(crack_area, 2),
        'percentage': round(percentage, 2),
        'position': {'x': 0, 'y': 0, 'w': 256, 'h': 256},
        'confidence': round(percentage / 100, 2),
        'severity': ["negligible", "minor", "moderate", "critical"][severity]
    }
    return f"/results/{result_filename}", [results] if severity > 0 else []

# Process video (unchanged)
def process_video(video_path, threshold=0.5, min_size=30):
    logger.info(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None, "Failed to open video"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    result_filename = f"result_{uuid.uuid4()}.mp4"
    result_path = os.path.join(RESULTS_FOLDER, result_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_path, fourcc, fps, (256, 256))
    frame_count = 0
    max_frames = 300
    all_results = []
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 5 == 0:
            image_tensor, enhanced_image = preprocess_image(frame)
            image_tensor = image_tensor.to(device)
            with torch.no_grad():
                logits = crack_model(image_tensor)
                probabilities = torch.sigmoid(logits)
                mask = probabilities.squeeze().cpu().numpy()
            processed_mask = advanced_post_processing(mask, min_size, threshold)
            result_text, percentage, contours, severity, crack_area = analyze_crack_severity(processed_mask)
            original = np.array(enhanced_image)
            result_image = original.copy()
            color_map = {0: (0, 255, 0), 1: (255, 255, 0), 2: (0, 165, 255), 3: (0, 0, 255)}
            color = color_map[severity]
            if contours:
                cv2.drawContours(result_image, contours, -1, color, 2)
            overlay = result_image.copy()
            for c in range(3):
                overlay[:,:,c] = np.where(processed_mask > 0, color[c], overlay[:,:,c])
            alpha = 0.3
            result_image = cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0)
            if severity > 0:
                result = {
                    'id': len(all_results) + 1,
                    'crack_area_cm2': round(crack_area, 2),
                    'percentage': round(percentage, 2),
                    'position': {'x': 0, 'y': 0, 'w': 256, 'h': 256},
                    'confidence': round(percentage / 100, 2),
                    'severity': ["negligible", "minor", "moderate", "critical"][severity]
                }
                all_results.append(result)
            out.write(cv2.resize(result_image, (256, 256)))
        else:
            resized_frame = cv2.resize(frame, (256, 256))
            out.write(resized_frame)
        frame_count += 1
    cap.release()
    out.release()
    return f"/results/{result_filename}", all_results

# Background processing
def background_process(file_path, file_type, threshold=0.5, min_size=30):
    sleep(1)  # Simulate upload delay
    if file_type == 'image':
        return process_image_async(file_path, threshold, min_size)
    elif file_type == 'video':
        return process_video(file_path, threshold, min_size)
    return None, "Unsupported file type"

@app.route('/api/upload', methods=['POST'])
def upload_file():
    logger.info("Received upload request")
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part in the request', 'progress': 0}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({'error': 'No file selected', 'progress': 0}), 400
    
    threshold = float(request.form.get('threshold', 0.5))
    min_size = int(request.form.get('min_size', 30))
    
    filename = f"upload_{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    try:
        file.save(filepath)
        logger.info(f"File saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        return jsonify({'error': f"Failed to save file: {str(e)}", 'progress': 0}), 500
    
    file_type = 'image' if filename.lower().endswith(('.png', '.jpg', '.jpeg')) else 'video' if filename.lower().endswith(('.mp4', '.avi')) else None
    if not file_type:
        logger.error(f"Unsupported file type: {filename}")
        return jsonify({'error': 'Unsupported file type', 'progress': 0}), 400

    task_id = str(uuid.uuid4())
    def process_task():
        result_url, results = background_process(filepath, file_type, threshold, min_size)
        if result_url:
            task_results[task_id] = {'result_url': result_url, 'results': results, 'message': f"{file_type.capitalize()} processed successfully"}
        else:
            task_results[task_id] = {'error': results}

    Thread(target=process_task).start()
    return jsonify({'task_id': task_id, 'progress': 10, 'message': 'Processing started'}), 202

@app.route('/api/status/<task_id>', methods=['GET'])
def check_status(task_id):
    if task_id in task_results:
        result = task_results.pop(task_id)
        if 'error' in result:
            return jsonify({'error': result['error'], 'progress': 100}), 500
        return jsonify({
            'result_url': result['result_url'],
            'results': result['results'],
            'progress': 100,
            'message': result['message']
        })
    return jsonify({'progress': 50, 'message': 'Processing...'}), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': crack_model is not None, 'error': None if crack_model else 'Model not loaded'})

@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path.startswith('api/') or path.startswith('results/'):
        return "Method Not Allowed", 405
    if path and os.path.exists(os.path.join('build', path)):
        return send_from_directory('build', path)
    return send_from_directory('build', 'index.html')

initialize()
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, threaded=True)