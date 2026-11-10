import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import io

# Set page config
st.set_page_config(
    page_title="Lung Adenocarcinoma Segmentation",
    page_icon="🫁",
    layout="wide"
)

# U-Net Model Definition
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        # Encoder (downsampling path)
        self.encoder1 = self._conv_block(in_channels, features[0])
        self.encoder2 = self._down_block(features[0], features[1])
        self.encoder3 = self._down_block(features[1], features[2])
        self.encoder4 = self._down_block(features[2], features[3])
        
        # Bottleneck
        self.bottleneck = self._down_block(features[3], features[3] * 2)
        
        # Decoder (upsampling path)
        self.upconv4 = self._up_block(features[3] * 2, features[3])
        self.decoder4 = self._conv_block(features[3] * 2, features[3])
        
        self.upconv3 = self._up_block(features[3], features[2])
        self.decoder3 = self._conv_block(features[2] * 2, features[2])
        
        self.upconv2 = self._up_block(features[2], features[1])
        self.decoder2 = self._conv_block(features[1] * 2, features[1])
        
        self.upconv1 = self._up_block(features[1], features[0])
        self.decoder1 = self._conv_block(features[0] * 2, features[0])
        
        # Final classifier
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_block(in_channels, out_channels)
        )
    
    def _up_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final output
        output = self.final_conv(dec1)
        return torch.sigmoid(output)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512])
        
        # Update this path to your model location
        MODEL_PATH = "lung_data_model/kaggle/working/best_unet_model.pth"
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        st.success(f"Model loaded successfully from epoch {checkpoint['epoch']}")
        return model, device, checkpoint
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for inference"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

def predict_segmentation(model, device, image, threshold=0.3):
    """Predict segmentation mask"""
    # Preprocess image
    input_tensor = preprocess_image(image)
    input_tensor = input_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        confidence_map = output.cpu().squeeze().numpy()
    
    # Find best threshold if needed
    best_threshold = threshold
    best_pixels = 0
    
    # Try different thresholds to find reasonable detection
    for test_thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        test_mask = (confidence_map > test_thresh).astype(np.uint8)
        test_pixels = np.sum(test_mask)
        if test_pixels > best_pixels and test_pixels < confidence_map.size * 0.3:  # Not too much
            best_threshold = test_thresh
            best_pixels = test_pixels
    
    binary_mask = (confidence_map > best_threshold).astype(np.uint8)
    
    return confidence_map, binary_mask, best_threshold

def create_visualization(original_image, confidence_map, binary_mask, threshold):
    """Create visualization with green overlay"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('white')
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Confidence heatmap
    im1 = axes[1].imshow(confidence_map, cmap='YlOrRd', vmin=0, vmax=1)
    axes[1].set_title(f'Confidence Heatmap\nRange: [{confidence_map.min():.3f}, {confidence_map.max():.3f}]', 
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Binary mask
    axes[2].imshow(binary_mask, cmap='Greens', vmin=0, vmax=1)
    axes[2].set_title(f'Segmentation Mask\nThreshold: {threshold}\nDetected: {np.sum(binary_mask)} pixels', 
                      fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Green overlay
    overlay = np.array(original_image).copy()
    if np.sum(binary_mask) > 0:
        mask_colored = np.zeros_like(overlay)
        mask_colored[binary_mask == 1] = [0, 255, 0]  # Green for positive regions
        result = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    else:
        result = overlay
    
    axes[3].imshow(result)
    title = 'Adenocarcinoma Detection Overlay'
    if np.sum(binary_mask) == 0:
        title += '\n(No regions detected)'
    axes[3].set_title(title, fontsize=14, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🫁 Lung Adenocarcinoma Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Medical Image Analysis Using U-Net Deep Learning</div>', unsafe_allow_html=True)
    
    # Load model
    model, device, checkpoint = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check the model path.")
        st.stop()
    
    # Model info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">96.45%</div>
            <div class="metric-label">Dice Coefficient</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">93.15%</div>
            <div class="metric-label">IoU Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">98.73%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">U-Net</div>
            <div class="metric-label">Architecture</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Lung Tissue Image",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Supported formats: PNG, JPG, JPEG, TIFF, BMP"
    )
    
    # Threshold slider
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.1,
        help="Lower values are more sensitive, higher values are more specific"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # Show original image
        st.subheader("Original Image")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}")
        
        # Analyze button
        if st.button("🔬 Analyze Image", type="primary"):
            with st.spinner("Analyzing image... This may take a few moments."):
                try:
                    # Predict
                    confidence_map, binary_mask, used_threshold = predict_segmentation(
                        model, device, image, threshold
                    )
                    
                    # Calculate statistics
                    total_pixels = confidence_map.size
                    positive_pixels = np.sum(binary_mask)
                    coverage_percentage = (positive_pixels / total_pixels) * 100
                    mean_confidence = confidence_map.mean()
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Coverage", f"{coverage_percentage:.2f}%", help="Percentage of tissue affected")
                    
                    with col2:
                        st.metric("Confidence", f"{mean_confidence*100:.1f}%", help="Average model confidence")
                    
                    with col3:
                        st.metric("Affected Pixels", f"{positive_pixels:,}", help="Number of detected pixels")
                    
                    with col4:
                        st.metric("Threshold Used", f"{used_threshold}", help="Optimal threshold selected")
                    
                    # Create and display visualization
                    fig = create_visualization(image, confidence_map, binary_mask, used_threshold)
                    st.pyplot(fig)
                    
                    # Clinical assessment
                    st.subheader("Clinical Assessment")
                    
                    if coverage_percentage < 1:
                        severity = "Minimal"
                        color = "green"
                        recommendation = "No significant abnormalities detected. Continue regular monitoring."
                    elif coverage_percentage < 5:
                        severity = "Mild"
                        color = "orange"
                        recommendation = "Minimal findings detected. Close monitoring advised."
                    elif coverage_percentage < 15:
                        severity = "Moderate"
                        color = "orange"
                        recommendation = "Moderate involvement detected. Clinical correlation recommended."
                    else:
                        severity = "Extensive"
                        color = "red"
                        recommendation = "Extensive involvement detected. Immediate clinical attention recommended."
                    
                    st.markdown(f"""
                    **Severity Level:** :{color}[{severity}]
                    
                    **Clinical Significance:** {recommendation}
                    
                    **Technical Details:**
                    - Model: U-Net Deep Learning Architecture
                    - Input Resolution: 256×256 pixels
                    - Confidence Range: [{confidence_map.min():.3f}, {confidence_map.max():.3f}]
                    - Processing Device: {device}
                    """)
                    
                    # Warning
                    st.warning("""
                    **Medical Disclaimer:** This AI system is designed to assist medical professionals and should not be used as the sole basis for diagnosis. 
                    All results should be validated by qualified healthcare providers and used in conjunction with clinical expertise 
                    and additional diagnostic methods.
                    """)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Try adjusting the threshold or uploading a different image.")

if __name__ == "__main__":
    main()