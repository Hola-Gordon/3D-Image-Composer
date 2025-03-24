import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

def load_segmentation_model():
    """Load a pre-trained DeepLabV3 model for person segmentation."""
    # Load pre-trained DeepLabV3 model
    model = deeplabv3_resnet50(pretrained=True)
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, device

def segment_person(image_path, model, device):
    """
    Extract a person from an image using semantic segmentation.
    
    Args:
        image_path: Path to the input image
        model: Pre-trained segmentation model
        device: Device to run the model on
        
    Returns:
        The segmented person with a transparent background
    """
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Prepare image for the model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    # The model outputs a multi-class segmentation mask
    # Class 15 corresponds to 'person' in COCO dataset
    person_mask = (output.argmax(0) == 15).cpu().numpy().astype(np.uint8)
    
    # Apply some morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)
    
    # Create a 4-channel RGBA image (RGB + Alpha)
    h, w = image.shape[:2]
    result = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Copy RGB channels
    result[:, :, :3] = image
    
    # Set alpha channel (transparent where there's no person)
    result[:, :, 3] = person_mask * 255
    
    return result