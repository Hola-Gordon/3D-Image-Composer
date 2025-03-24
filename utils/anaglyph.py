import cv2
import numpy as np

def create_anaglyph(left_img, right_img):
    """
    Create an anaglyph image from left and right stereoscopic images.
    
    Args:
        left_img: Left image (will be encoded in red channel)
        right_img: Right image (will be encoded in green and blue channels)
        
    Returns:
        Anaglyph image viewable with red-cyan glasses
    """
    # Convert images to the RGB color space if they're in BGR
    if left_img.shape[2] == 3:
        left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    else:
        left_img_rgb = left_img
        right_img_rgb = right_img
    
    # Create an empty image with the same dimensions
    anaglyph = np.zeros_like(left_img_rgb)
    
    # Extract channels
    # For left image: red channel
    left_red = left_img_rgb[:, :, 0]
    
    # For right image: green and blue channels
    right_green = right_img_rgb[:, :, 1]
    right_blue = right_img_rgb[:, :, 2]
    
    # Combine channels to create anaglyph
    anaglyph[:, :, 0] = left_red      # Red channel from left image
    anaglyph[:, :, 1] = right_green   # Green channel from right image
    anaglyph[:, :, 2] = right_blue    # Blue channel from right image
    
    # Convert back to BGR for OpenCV functions
    anaglyph_bgr = cv2.cvtColor(anaglyph, cv2.COLOR_RGB2BGR)
    
    return anaglyph_bgr

def visualize_results(original_image, left_image, right_image, anaglyph):
    """
    Visualize the original stereo pair, individual left/right images, and the resulting anaglyph.
    
    Args:
        original_image: Original side-by-side stereoscopic image
        left_image: Left image
        right_image: Right image
        anaglyph: Generated anaglyph image
    """
    import matplotlib.pyplot as plt
    
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    left_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
    anaglyph_rgb = cv2.cvtColor(anaglyph, cv2.COLOR_BGR2RGB)
    
    # Create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot original stereoscopic image
    axs[0, 0].imshow(original_rgb)
    axs[0, 0].set_title('Original Stereoscopic Image')
    axs[0, 0].axis('off')
    
    # Plot left image
    axs[0, 1].imshow(left_rgb)
    axs[0, 1].set_title('Left Image')
    axs[0, 1].axis('off')
    
    # Plot right image
    axs[1, 0].imshow(right_rgb)
    axs[1, 0].set_title('Right Image')
    axs[1, 0].axis('off')
    
    # Plot anaglyph
    axs[1, 1].imshow(anaglyph_rgb)
    axs[1, 1].set_title('Anaglyph Image (Use Red-Cyan Glasses)')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()