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

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get port from environment variable (for Render.com)
port = int(os.environ.get("PORT", 5000))

# Initialize Flask app
app = Flask(__name__, 
    static_folder='build',  # Serve static files from backend/build
    template_folder='build') # Serve HTML from backend/build
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
MODELS_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Global model variables
crack_model = None

# Model Architecture (unchanged from your code)
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
            # Check file size to make sure it's valid
            file_size = os.path.getsize(model_path)
            logger.info(f"Model file size: {file_size / (1024*1024):.2f} MB")
            
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
    # Check build folder
    if not os.path.exists('build'):
        logger.warning("Build folder not found! Frontend assets might be missing.")
    else:
        if os.path.exists('build/index.html'):
            logger.info("Frontend build found and ready")
        else:
            logger.warning("index.html not found in build folder!")
    
    # Check if model file exists
    model_path = os.path.join(MODELS_FOLDER, 'crack_detector_weights.pth')
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}")
    else:
        logger.info(f"Model file found at {model_path} with size {os.path.getsize(model_path)} bytes")
    
    # Load models
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
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(enhanced_image).unsqueeze(0), enhanced_image

# Post-processing
def advanced_post_processing(mask, min_size=30, threshold=0.5):
    binary_mask = (mask > threshold).astype(np.uint8)
    cleaned = remove_small_objects(binary_mask.astype(bool), min_size=min_size)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed

# Analyze crack severity
def analyze_crack_severity(mask):
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
    image = cv2.imread(image_path)
    if image is None:
        return None, "Failed to load image"
    
    if crack_model is None:
        return None, "Model not loaded"
    
    image_tensor, enhanced_image = preprocess_image(image)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = crack_model(image_tensor)
        probabilities = torch.sigmoid(logits)
        mask = probabilities.squeeze().cpu().numpy()
    
    processed_mask = advanced_post_processing(mask, min_size, threshold)
    
    result_text, percentage, contours, severity = analyze_crack_severity(processed_mask)
    
    original = np.array(enhanced_image.resize((256, 256)))
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
    
    result_filename = f"result_{uuid.uuid4()}.jpg"
    result_path = os.path.join(RESULTS_FOLDER, result_filename)
    
    cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    
    results = {
        'id': 1,
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
    
    return f"/results/{result_filename}", [results] if severity > 0 else []

# Process video
def process_video(video_path, threshold=0.5, min_size=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
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
            
            result_text, percentage, contours, severity = analyze_crack_severity(processed_mask)
            
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
            
            out.write(cv2.resize(result_image, (256, 256)))
        else:
            resized_frame = cv2.resize(frame, (256, 256))
            out.write(resized_frame)
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    return f"/results/{result_filename}", all_results

# Change this part in your app.py
# Replace the existing routes with these simplified ones

# API Routes
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    # Add simple health check endpoint for debugging
    if path == "health":
        return jsonify({"status": "healthy"})
        
    # For API endpoints
    if path.startswith('api/'):
        # These will be handled by their specific route handlers
        return app.view_functions[path.split('/')[-1]](path)
        
    # For result images
    if path.startswith('results/'):
        return send_from_directory('static/results', path.replace('results/', ''))
        
    # For other static files (JS, CSS, etc.)
    if path and os.path.exists(os.path.join('build', path)):
        return send_from_directory('build', path)
        
    # Default: serve index.html for client-side routing
    return send_from_directory('build', 'index.html')

# Initialize the app
initialize()

# Gunicorn configuration for Render
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)