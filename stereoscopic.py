import cv2
import numpy as np
import os

class StereoProcessor:
    def __init__(self):
        """
        Handle stereoscopic image processing
        """
        # Define disparity values for different depth levels
        self.depth_disparities = {
            'close': 50,    # Larger disparity = closer to viewer
            'medium': 25,   # Medium distance
            'far': 10       # Small disparity = further away
        }
    
    def create_stereo_pair(self, width=640, height=480):
        """
        Create a simple stereoscopic image pair with depth cues
        
        Args:
            width (int): Width of the generated images
            height (int): Height of the generated images
            
        Returns:
            tuple: (left_image, right_image) as numpy arrays in RGB format
        """
        # Create a simple gradient background
        left = np.zeros((height, width, 3), dtype=np.uint8)
        right = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a gradient background
        for i in range(height):
            color = int(220 * i / height) + 35
            left[i, :] = [color, color, color]
            right[i, :] = [color, color, color]
        
        # Add grid lines for depth perception
        for x in range(0, width, 50):
            cv2.line(left, (x, 0), (x, height), (150, 150, 150), 1)
            cv2.line(right, (x, 0), (x, height), (150, 150, 150), 1)
        
        for y in range(0, height, 50):
            cv2.line(left, (0, y), (width, y), (150, 150, 150), 1)
            cv2.line(right, (0, y), (width, y), (150, 150, 150), 1)
        
        # Add some basic shapes at different depths
        # Far shapes (small disparity)
        cv2.rectangle(left, (100, 100), (200, 200), (0, 0, 255), -1)
        cv2.rectangle(right, (105, 100), (205, 200), (0, 0, 255), -1)
        
        # Medium shapes
        cv2.circle(left, (width//2, height//2), 50, (0, 255, 0), -1)
        cv2.circle(right, (width//2 + 15, height//2), 50, (0, 255, 0), -1)
        
        # Close shapes (large disparity)
        cv2.circle(left, (width-150, height-150), 30, (255, 0, 0), -1)
        cv2.circle(right, (width-120, height-150), 30, (255, 0, 0), -1)
        
        return left, right
    
    def load_stereo_pair(self, left_path, right_path):
        """
        Load a stereoscopic image pair, or create a new one if files don't exist
        
        Args:
            left_path (str): Path to left image
            right_path (str): Path to right image
            
        Returns:
            tuple: (left_image, right_image) as numpy arrays in RGB format
        """
        if os.path.exists(left_path) and os.path.exists(right_path):
            # Load images
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)
            
            # Convert to RGB
            left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            
            return left_rgb, right_rgb
        else:
            # Create new stereo pair
            print(f"Stereo pair not found. Creating a new pair.")
            return self.create_stereo_pair()
    
    def insert_person(self, person_rgba, left_img, right_img, depth="medium", position_x=0.5, position_y=0.7, scale=1.0):
        """
        Insert a segmented person into a stereoscopic image pair at the specified depth
        
        Args:
            person_rgba (numpy.ndarray): Segmented person with alpha channel
            left_img (numpy.ndarray): Left stereoscopic image (RGB)
            right_img (numpy.ndarray): Right stereoscopic image (RGB)
            depth (str): Depth level - 'close', 'medium', or 'far'
            position_x (float): Horizontal position (0-1)
            position_y (float): Vertical position (0-1)
            scale (float): Scale factor for the person
            
        Returns:
            tuple: (modified_left, modified_right) with person inserted
        """
        # Scale if needed
        if scale != 1.0:
            new_width = int(person_rgba.shape[1] * scale)
            new_height = int(person_rgba.shape[0] * scale)
            person_rgba = cv2.resize(person_rgba, (new_width, new_height))
        
        # Get image dimensions
        h, w = left_img.shape[:2]
        ph, pw = person_rgba.shape[:2]
        
        # Calculate base coordinates
        x = int(w * position_x - pw / 2)
        y = int(h * position_y - ph)
        
        # Ensure within bounds
        x = max(0, min(x, w - pw))
        y = max(0, min(y, h - ph))
        
        # Get disparity for the selected depth
        disparity = self.depth_disparities.get(depth, self.depth_disparities["medium"])
        
        # Calculate left and right positions based on disparity
        left_x = x - disparity // 2
        right_x = x + disparity // 2
        
        # Ensure positions are within bounds
        left_x = max(0, min(left_x, w - pw))
        right_x = max(0, min(right_x, w - pw))
        
        # Create copies to avoid modifying originals
        left_modified = left_img.copy()
        right_modified = right_img.copy()
        
        # Overlay the person on both images
        self._overlay_image(left_modified, person_rgba, left_x, y)
        self._overlay_image(right_modified, person_rgba, right_x, y)
        
        return left_modified, right_modified
    
    def _overlay_image(self, background, foreground, x, y):
        """
        Overlay foreground image with transparency on background image
        
        Args:
            background (numpy.ndarray): Background image (modified in-place)
            foreground (numpy.ndarray): Foreground image with alpha channel
            x (int): X position to place foreground
            y (int): Y position to place foreground
        """
        # Get dimensions
        h, w = foreground.shape[:2]
        
        # Ensure we don't go out of bounds
        y_end = min(y + h, background.shape[0])
        x_end = min(x + w, background.shape[1])
        h = y_end - y
        w = x_end - x
        
        if h <= 0 or w <= 0:
            return  # Out of bounds
        
        # Get alpha channel and convert to float (0-1)
        alpha = foreground[:h, :w, 3] / 255.0
        alpha = alpha.reshape(h, w, 1)
        
        # Apply alpha blending
        background[y:y_end, x:x_end] = (
            foreground[:h, :w, :3] * alpha + 
            background[y:y_end, x:x_end] * (1 - alpha)
        ).astype(np.uint8)

# Example usage
if __name__ == "__main__":
    from segmentation import PersonSegmenter
    
    # Initialize processors
    stereo = StereoProcessor()
    segmenter = PersonSegmenter()
    
    # Create stereo pair
    left, right = stereo.create_stereo_pair()
    
    # Create default person
    person = segmenter.create_default_person()
    
    # For each depth level
    for depth in ['close', 'medium', 'far']:
        # Insert at different depths
        left_with_person, right_with_person = stereo.insert_person(
            person, left, right, depth=depth
        )
        
        # Save the results
        cv2.imwrite(f"stereo_{depth}_left.jpg", cv2.cvtColor(left_with_person, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"stereo_{depth}_right.jpg", cv2.cvtColor(right_with_person, cv2.COLOR_RGB2BGR))
        
        print(f"Created stereoscopic pair with person at {depth} depth")
    
    print("Stereo processing complete!")