import cv2
import numpy as np
import os

class AnaglyphGenerator:
    def __init__(self):
        """
        Simple anaglyph generation for red-cyan 3D glasses
        """
        pass
    
    def create_anaglyph(self, left_img, right_img, method='true'):
        """
        Create an anaglyph image from a stereoscopic pair
        
        Args:
            left_img (numpy.ndarray): Left image in RGB format
            right_img (numpy.ndarray): Right image in RGB format
            method (str): Anaglyph method - 'true', 'optimized', or 'gray'
            
        Returns:
            numpy.ndarray: Anaglyph image in RGB format
        """
        # Ensure RGB format
        if left_img.shape[2] == 3 and left_img[0,0,0] == left_img[0,0,2]:  # Check if B==R (usually means BGR)
            left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        else:
            left_rgb = left_img.copy()
            right_rgb = right_img.copy()
        
        # Split into color channels
        left_r, left_g, left_b = cv2.split(left_rgb)
        right_r, right_g, right_b = cv2.split(right_rgb)
        
        if method == 'true':
            # True anaglyph (red from left eye, cyan from right eye)
            # This gives good depth but poor color reproduction
            anaglyph_r = left_r
            anaglyph_g = right_g
            anaglyph_b = right_b
        
        elif method == 'optimized':
            # Optimized anaglyph (preserves more color but may have ghosting)
            anaglyph_r = left_r
            anaglyph_g = 0.7 * right_g + 0.3 * left_g
            anaglyph_b = right_b
        
        elif method == 'gray':
            # Gray anaglyph (monochrome version for less eye strain)
            left_gray = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
            
            anaglyph_r = left_gray
            anaglyph_g = right_gray
            anaglyph_b = right_gray
        
        else:
            # Default to true anaglyph
            anaglyph_r = left_r
            anaglyph_g = right_g
            anaglyph_b = right_b
        
        # Merge color channels
        anaglyph = cv2.merge([anaglyph_r, anaglyph_g, anaglyph_b])
        
        return anaglyph
    
    def optimize_anaglyph(self, anaglyph, brightness=1.0, contrast=1.0):
        """
        Enhance anaglyph image appearance
        
        Args:
            anaglyph (numpy.ndarray): Anaglyph image
            brightness (float): Brightness adjustment (0.5-1.5)
            contrast (float): Contrast adjustment (0.5-1.5)
            
        Returns:
            numpy.ndarray: Enhanced anaglyph image
        """
        # Apply contrast and brightness
        # Formula: new_img = contrast * img + brightness
        adjusted = cv2.convertScaleAbs(
            anaglyph, 
            alpha=contrast, 
            beta=(brightness - 1.0) * 50
        )
        
        return adjusted

# Example usage
if __name__ == "__main__":
    from stereoscopic import StereoProcessor
    from segmentation import PersonSegmenter
    
    # Initialize components
    stereo = StereoProcessor()
    anaglyph_gen = AnaglyphGenerator()
    segmenter = PersonSegmenter()
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Create stereo pair
    left, right = stereo.create_stereo_pair()
    
    # Create default person
    person = segmenter.create_default_person()
    
    # Insert person at medium depth
    left_with_person, right_with_person = stereo.insert_person(
        person, left, right, depth="medium"
    )
    
    # Create anaglyphs with different methods
    for method in ['true', 'optimized', 'gray']:
        # Create anaglyph
        anaglyph = anaglyph_gen.create_anaglyph(
            left_with_person, right_with_person, method=method
        )
        
        # Optimize appearance
        anaglyph = anaglyph_gen.optimize_anaglyph(anaglyph)
        
        # Save the result
        cv2.imwrite(f"output/anaglyph_{method}.jpg", cv2.cvtColor(anaglyph, cv2.COLOR_RGB2BGR))
        
        print(f"Created {method} anaglyph")
    
    print("Anaglyph processing complete!")