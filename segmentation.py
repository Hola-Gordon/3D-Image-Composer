import cv2
import numpy as np
import os
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image

class PersonSegmenter:
    def __init__(self):
        """
        Segmentation class using pretrained Mask R-CNN to extract people
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pretrained Mask R-CNN model
        try:
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            print("Loaded Mask R-CNN model successfully")
            
            # Person class index in COCO dataset (used by Mask R-CNN)
            self.person_class_index = 1
        except Exception as e:
            print(f"Error loading Mask R-CNN: {e}")
            self.model = None
    
    def segment_person(self, image_np):
        """
        Segment person from image using Mask R-CNN
        
        Args:
            image_np (numpy.ndarray): Input image (RGB or BGR)
            
        Returns:
            numpy.ndarray: RGBA image with background made transparent
        """
        # Convert to RGB format
        if image_np.shape[2] == 3:
            if image_np[0,0,0] > image_np[0,0,2]:  # Likely BGR format
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_np.copy()
        else:
            image_rgb = cv2.cvtColor(image_np[:,:,:3], cv2.COLOR_RGBA2RGB)
        
        # Create empty mask the same size as the image
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Use the model to detect and segment people
        if self.model is not None:
            try:
                # Convert to PIL Image and then process
                pil_image = Image.fromarray(image_rgb)
                
                # Transform for the model
                image_tensor = F.to_tensor(pil_image).to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    predictions = self.model([image_tensor])
                
                # Process predictions
                if len(predictions) > 0:
                    prediction = predictions[0]
                    
                    # Extract masks for persons only
                    person_indices = [i for i, label in enumerate(prediction['labels']) 
                                     if label.item() == self.person_class_index]
                    
                    if person_indices:
                        # Use the mask with highest score
                        best_idx = max(person_indices, key=lambda i: prediction['scores'][i].item())
                        
                        # Get mask for best person
                        mask_tensor = prediction['masks'][best_idx, 0]
                        mask_np = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8) * 255
                        
                        # Resize mask to match the original image
                        mask = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    else:
                        print("No person detected. Using default silhouette.")
                        mask = self._create_default_mask(h, w)
                else:
                    print("No predictions. Using default silhouette.")
                    mask = self._create_default_mask(h, w)
            except Exception as e:
                print(f"Error in segmentation: {e}")
                mask = self._create_default_mask(h, w)
        else:
            # If model failed to load, use default silhouette
            print("Using default silhouette (no model)")
            mask = self._create_default_mask(h, w)
        
        # Create RGBA with transparency from mask
        rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = mask
        
        return rgba
    
    def _create_default_mask(self, height, width):
        """
        Create a default person silhouette mask
        
        Args:
            height (int): Height of the mask
            width (int): Width of the mask
            
        Returns:
            numpy.ndarray: Mask with person silhouette
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create a simple person silhouette
        # Head
        head_radius = min(width, height) // 6
        head_center = (width // 2, height // 4)
        cv2.circle(mask, head_center, head_radius, 255, -1)
        
        # Body
        body_width = int(width * 0.4)
        body_top = head_center[1] + head_radius - 10
        body_left = width // 2 - body_width // 2
        body_height = int(height * 0.55)
        cv2.rectangle(mask, (body_left, body_top), 
                     (body_left + body_width, body_top + body_height), 255, -1)
        
        # Legs
        leg_width = body_width // 3
        leg_top = body_top + body_height
        leg_height = height - leg_top
        
        # Left leg
        cv2.rectangle(mask, (body_left + leg_width // 2, leg_top),
                     (body_left + leg_width * 3 // 2, height), 255, -1)
        
        # Right leg
        cv2.rectangle(mask, (body_left + body_width - leg_width * 3 // 2, leg_top),
                     (body_left + body_width - leg_width // 2, height), 255, -1)
        
        # Arms
        arm_width = leg_width
        arm_top = body_top + body_height // 4
        arm_height = body_height // 3
        
        # Left arm
        cv2.rectangle(mask, (body_left - arm_width, arm_top),
                     (body_left, arm_top + arm_height), 255, -1)
        
        # Right arm
        cv2.rectangle(mask, (body_left + body_width, arm_top),
                     (body_left + body_width + arm_width, arm_top + arm_height), 255, -1)
        
        return mask
    
    def create_default_person(self, width=300, height=400):
        """
        Create a default person silhouette
        
        Args:
            width (int): Width of the image
            height (int): Height of the image
            
        Returns:
            numpy.ndarray: RGBA image with person silhouette
        """
        # Create a light gray background
        image = np.ones((height, width, 3), dtype=np.uint8) * 210
        
        # Create mask with person silhouette
        mask = self._create_default_mask(height, width)
        
        # Draw the silhouette in a darker gray
        image[mask > 0] = [120, 120, 120]
        
        # Create RGBA with full opacity for person pixels
        rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = mask
        
        return rgba

# Example usage
if __name__ == "__main__":
    segmenter = PersonSegmenter()
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Test with a default person
    default_person = segmenter.create_default_person()
    cv2.imwrite("output/default_person.png", cv2.cvtColor(default_person, cv2.COLOR_RGBA2BGRA))
    
    # Test with an image if it exists
    test_image_path = "test_person.jpg"
    if os.path.exists(test_image_path):
        test_image = cv2.imread(test_image_path)
        segmented = segmenter.segment_person(test_image)
        cv2.imwrite("output/segmented_person.png", cv2.cvtColor(segmented, cv2.COLOR_RGBA2BGRA))
    
    print("Segmentation testing complete!")