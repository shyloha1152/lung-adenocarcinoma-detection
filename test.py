import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Define the complete U-Net model class
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

def load_model(model_path, device):
    """Load the trained U-Net model from checkpoint."""
    # Create model instance
    model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512])
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from epoch {checkpoint['epoch']}")
    print(f"Best validation Dice score: {checkpoint['val_dice']:.4f}")
    
    return model

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess image for inference - must match training preprocessing.
    
    Args:
        image_path (str): Path to input image
        target_size (tuple): Target image size
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Define preprocessing transforms (same as training)
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms and add batch dimension
    input_tensor = transform(image).unsqueeze(0)
    
    return input_tensor, image

def predict_mask(model, image_tensor, device, threshold=0.5):
    """
    Predict segmentation mask for input image.
    
    Args:
        model: Trained U-Net model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        threshold (float): Threshold for binary segmentation
    
    Returns:
        tuple: (confidence_map, binary_mask)
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Forward pass
        output = model(image_tensor)
        
        # Convert to numpy
        confidence_map = output.cpu().squeeze().numpy()
        
        # Apply threshold for binary mask
        binary_mask = (confidence_map > threshold).astype(np.uint8)
        
    return confidence_map, binary_mask

def visualize_results(original_image, confidence_map, binary_mask, threshold=0.5, save_path=None):
    """
    Visualize inference results.
    
    Args:
        original_image: Original PIL image
        confidence_map: Model confidence output (0-1)
        binary_mask: Binary segmentation mask
        threshold (float): Threshold used for binarization
        save_path (str): Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Confidence map
    im1 = axes[1].imshow(confidence_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(f'Confidence Map\nRange: [{confidence_map.min():.3f}, {confidence_map.max():.3f}]', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Binary mask
    axes[2].imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Binary Mask (threshold={threshold})\nPositive pixels: {np.sum(binary_mask)}', fontsize=14)
    axes[2].axis('off')
    
    # Overlay
    overlay = np.array(original_image).copy()
    mask_colored = np.zeros_like(overlay)
    mask_colored[binary_mask == 1] = [255, 0, 0]  # Red for positive regions
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (Red = Predicted Mask)', fontsize=14)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to: {save_path}")
    
    plt.show()

def analyze_prediction(confidence_map, binary_mask, threshold=0.5):
    """
    Analyze prediction statistics.
    
    Args:
        confidence_map: Model confidence output
        binary_mask: Binary mask
        threshold: Threshold used
    """
    total_pixels = confidence_map.size
    positive_pixels = np.sum(binary_mask)
    negative_pixels = total_pixels - positive_pixels
    
    print("Prediction Analysis:")
    print(f"Image size: {confidence_map.shape}")
    print(f"Total pixels: {total_pixels}")
    print(f"Confidence range: [{confidence_map.min():.4f}, {confidence_map.max():.4f}]")
    print(f"Mean confidence: {confidence_map.mean():.4f}")
    print(f"Threshold used: {threshold}")
    print(f"Positive pixels: {positive_pixels} ({positive_pixels/total_pixels*100:.2f}%)")
    print(f"Negative pixels: {negative_pixels} ({negative_pixels/total_pixels*100:.2f}%)")
    
    # Check if prediction is all black or all white
    if positive_pixels == 0:
        print("WARNING: Prediction is all black (no positive pixels)")
        print("Try lowering the threshold or check your preprocessing")
    elif positive_pixels == total_pixels:
        print("WARNING: Prediction is all white (all positive pixels)")
        print("Try increasing the threshold")

def inference_pipeline(model_path, image_path, threshold=0.5, save_results=None):
    """
    Complete inference pipeline.
    
    Args:
        model_path (str): Path to saved model checkpoint
        image_path (str): Path to input image
        threshold (float): Segmentation threshold
        save_results (str): Path to save results (optional)
    
    Returns:
        tuple: (confidence_map, binary_mask)
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    
    # Preprocess image
    print("Preprocessing image...")
    image_tensor, original_image = preprocess_image(image_path)
    print(f"Input tensor shape: {image_tensor.shape}")
    
    # Predict
    print("Running inference...")
    confidence_map, binary_mask = predict_mask(model, image_tensor, device, threshold)
    
    # Analyze results
    analyze_prediction(confidence_map, binary_mask, threshold)
    
    # Visualize
    visualize_results(original_image, confidence_map, binary_mask, threshold, save_results)
    
    return confidence_map, binary_mask

# Example usage
if __name__ == "__main__":
    # Update these paths according to your setup
    MODEL_PATH = "lung_data_model/kaggle/working/best_unet_model.pth"  # Update this path
    IMAGE_PATH = r"C:\Users\praga\Documents\Projects\lungs_seg\lung_data_model\kaggle\working\test\images\patch_x36006_y52176.png"       # Update this path
    
    # Run inference
    confidence_map, binary_mask = inference_pipeline(
        model_path=MODEL_PATH,
        image_path=IMAGE_PATH,
        threshold=0.5,
        save_results="inference_results.png"
    )
    
    # If results are all black, try different thresholds
    if np.sum(binary_mask) == 0:
        print("\nTrying lower thresholds...")
        for thresh in [0.3, 0.1, 0.05]:
            print(f"\nTesting threshold: {thresh}")
            _, test_mask = predict_mask(
                load_model(MODEL_PATH, torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                *preprocess_image(IMAGE_PATH),
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                thresh
            )
            print(f"Positive pixels with threshold {thresh}: {np.sum(test_mask)}")