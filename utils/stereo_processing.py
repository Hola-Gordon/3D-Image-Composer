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

def insert_person_with_depth_position(left_bg, right_bg, person_img, depth_level='medium', 
                                     x_position=50, y_position=75, scale_factor=0.67):
    """
    Insert a segmented person into stereoscopic images with controllable depth, position, and scale.
    
    Args:
        left_bg: Left background image
        right_bg: Right background image
        person_img: Segmented person image with alpha channel
        depth_level: Desired depth ('close', 'medium', or 'far')
        x_position: Horizontal position as percentage (0-100) of background width
        y_position: Vertical position as percentage (0-100) of background height
        scale_factor: Size scale factor relative to original size
        
    Returns:
        Tuple of (left_composite, right_composite)
    """
    # Get dimensions
    bg_height, bg_width = left_bg.shape[:2]
    person_height, person_width = person_img.shape[:2]
    
    # Resize person based on scale factor
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
    
    # Calculate center position based on percentage inputs
    # Convert from percentage to pixel coordinates
    center_x = int((x_position / 100) * bg_width)
    center_y = int((y_position / 100) * bg_height)
    
    # Calculate the top-left corner of the person image
    base_x = center_x - new_width // 2
    base_y = center_y - new_height // 2
    
    # Calculate positions for left and right images with disparity offset
    left_x = base_x + disparity // 2
    right_x = base_x - disparity // 2
    
    # Create copies of the background images for the composites
    left_composite = left_bg.copy()
    right_composite = right_bg.copy()
    
    # Extract alpha channel from the resized person image
    alpha = person_resized[:, :, 3] / 255.0
    
    # Blend function for safer compositing (handles out-of-bounds regions)
    def blend_with_boundary_check(bg, fg, x, y, alpha_channel):
        # Get dimensions
        fg_h, fg_w = fg.shape[:2]
        bg_h, bg_w = bg.shape[:2]
        
        # Calculate the valid region to blend
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(x + fg_w, bg_w)
        y_end = min(y + fg_h, bg_h)
        
        # If there's no valid region, return the background unchanged
        if x_start >= x_end or y_start >= y_end:
            return bg
        
        # Calculate offsets for the foreground image
        fg_x_start = x_start - x
        fg_y_start = y_start - y
        fg_x_end = fg_x_start + (x_end - x_start)
        fg_y_end = fg_y_start + (y_end - y_start)
        
        # Get the valid region from the foreground and its alpha
        fg_valid = fg[fg_y_start:fg_y_end, fg_x_start:fg_x_end]
        alpha_valid = alpha_channel[fg_y_start:fg_y_end, fg_x_start:fg_x_end]
        
        # Create a copy of the background to avoid modifying the original
        result = bg.copy()
        
        # Blend the valid region
        for c in range(3):  # For each color channel
            result[y_start:y_end, x_start:x_end, c] = (
                result[y_start:y_end, x_start:x_end, c] * (1 - alpha_valid) + 
                fg_valid[:, :, c] * alpha_valid
            )
        
        return result
    
    # Perform the blending for both left and right images
    left_composite = blend_with_boundary_check(left_composite, person_resized, left_x, base_y, alpha)
    right_composite = blend_with_boundary_check(right_composite, person_resized, right_x, base_y, alpha)
    
    return left_composite, right_composite