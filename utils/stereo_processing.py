"""
Stereoscopic image processing functions for 3D image composition.
"""

import cv2
import numpy as np

def load_stereo_pair(image_path):
    """
    Load a side-by-side stereoscopic image pair and split it into left and right images.
    
    Args:
        image_path: Path to the side-by-side stereoscopic image
        
    Returns:
        Tuple of (left_image, right_image)
    """
    # Read the stereoscopic image
    stereo_image = cv2.imread(image_path)
    
    # Get dimensions
    height, width = stereo_image.shape[:2]
    midpoint = width // 2
    
    # Split into left and right images
    left_image = stereo_image[:, :midpoint]
    right_image = stereo_image[:, midpoint:]
    
    return left_image, right_image

def insert_person_with_depth(left_bg, right_bg, person_img, depth_level='medium'):
    """
    Insert a segmented person into the left and right stereoscopic images with the specified depth.
    
    Args:
        left_bg: Left background image
        right_bg: Right background image
        person_img: Segmented person image with alpha channel
        depth_level: Desired depth ('close', 'medium', or 'far')
        
    Returns:
        Tuple of (left_composite, right_composite)
    """
    # Resize person to a reasonable size relative to the background
    bg_height, bg_width = left_bg.shape[:2]
    person_height, person_width = person_img.shape[:2]
    
    # Scale person to be about 2/3 of the background height
    scale_factor = (bg_height * 0.67) / person_height
    new_width = int(person_width * scale_factor)
    new_height = int(person_height * scale_factor)
    person_resized = cv2.resize(person_img, (new_width, new_height))
    
    # Define disparity values for different depth levels
    # Larger disparity makes objects appear closer
    disparities = {
        'close': 40,      # Large disparity for close objects
        'medium': 20,     # Medium disparity
        'far': 5          # Small disparity for distant objects
    }
    
    disparity = disparities.get(depth_level, 20)  # Default to medium if invalid
    
    # Calculate positions for left and right images
    # For the left image, shift right, for the right image, shift left
    left_x = (bg_width - new_width) // 2 + disparity // 2
    right_x = (bg_width - new_width) // 2 - disparity // 2
    y = bg_height - new_height
    
    # Create copies of the background images for the composites
    left_composite = left_bg.copy()
    right_composite = right_bg.copy()
    
    # Extract alpha channel from the resized person image
    alpha = person_resized[:, :, 3] / 255.0
    
    # Define the regions where the person will be inserted
    left_roi = left_composite[y:y+new_height, left_x:left_x+new_width]
    right_roi = right_composite[y:y+new_height, right_x:right_x+new_width]
    
    # Ensure ROI is within the image bounds
    if left_x >= 0 and left_x + new_width <= bg_width and y >= 0 and y + new_height <= bg_height:
        # Blend the person with the background for the left image
        for c in range(3):  # For each color channel
            left_roi[:, :, c] = left_roi[:, :, c] * (1 - alpha) + person_resized[:, :, c] * alpha
    
    # Ensure ROI is within the image bounds
    if right_x >= 0 and right_x + new_width <= bg_width and y >= 0 and y + new_height <= bg_height:
        # Blend the person with the background for the right image
        for c in range(3):  # For each color channel
            right_roi[:, :, c] = right_roi[:, :, c] * (1 - alpha) + person_resized[:, :, c] * alpha
    
    return left_composite, right_composite