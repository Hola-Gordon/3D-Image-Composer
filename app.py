import os
import cv2
import numpy as np
import gradio as gr
from PIL import Image
import tempfile

# Import our modules
from segmentation import PersonSegmenter
from stereoscopic import StereoProcessor
from anaglyph import AnaglyphGenerator

# Create output directory
os.makedirs("output", exist_ok=True)

# Initialize components
segmenter = PersonSegmenter()
stereo_processor = StereoProcessor()
anaglyph_generator = AnaglyphGenerator()

def process_image(input_image, depth, position_x, position_y, scale, anaglyph_method, brightness, contrast):
    """
    Main processing function for the Gradio interface
    
    Args:
        input_image: Image uploaded by user (or None)
        depth: Depth level (close, medium, far)
        position_x: Horizontal position (0-1)
        position_y: Vertical position (0-1)
        scale: Scale factor for the person
        anaglyph_method: Anaglyph creation method
        brightness: Brightness adjustment
        contrast: Contrast adjustment
        
    Returns:
        list: [segmented_person, left_image, right_image, anaglyph]
    """
    # Process person image
    if input_image is None:
        # If no image provided, use default person silhouette
        person_rgba = segmenter.create_default_person()
    else:
        # Segment the uploaded image
        person_rgba = segmenter.segment_person(input_image)
    
    # Create stereo pair
    left_img, right_img = stereo_processor.create_stereo_pair()
    
    # Insert person at specified depth
    left_with_person, right_with_person = stereo_processor.insert_person(
        person_rgba, left_img, right_img, 
        depth=depth, position_x=position_x, position_y=position_y, scale=scale
    )
    
    # Create anaglyph
    anaglyph = anaglyph_generator.create_anaglyph(
        left_with_person, right_with_person, method=anaglyph_method
    )
    
    # Optimize anaglyph appearance
    anaglyph = anaglyph_generator.optimize_anaglyph(
        anaglyph, brightness=brightness, contrast=contrast
    )
    
    # Save outputs to files
    cv2.imwrite(f"output/segmented_person.png", cv2.cvtColor(person_rgba, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(f"output/stereo_{depth}_left.jpg", cv2.cvtColor(left_with_person, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"output/stereo_{depth}_right.jpg", cv2.cvtColor(right_with_person, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"output/anaglyph_{depth}_{anaglyph_method}.jpg", cv2.cvtColor(anaglyph, cv2.COLOR_RGB2BGR))
    
    # Convert RGBA to RGB for display in Gradio
    person_rgb = np.ones((person_rgba.shape[0], person_rgba.shape[1], 3), dtype=np.uint8) * 255
    mask = person_rgba[:,:,3] > 0
    person_rgb[mask] = person_rgba[mask, :3]
    
    return [person_rgb, left_with_person, right_with_person, anaglyph]

# Create Gradio interface
with gr.Blocks(title="3D Image Composer") as app:
    gr.Markdown("# 3D Image Composer")
    gr.Markdown("""
    This application allows you to insert a person into a stereoscopic 3D scene and create an anaglyph image.
    
    1. Upload an image with a person (or use the default silhouette)
    2. Adjust depth, position, and appearance settings
    3. Click "Process Image" to see the results
    4. View the anaglyph with red-cyan 3D glasses
    
    *All outputs are also saved to the 'output' folder*
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Person Image (optional)", type="numpy")
            
            with gr.Accordion("Depth & Position", open=True):
                depth = gr.Radio(
                    choices=["close", "medium", "far"], 
                    label="Depth Level", 
                    value="medium"
                )
                position_x = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.5, 
                    label="Horizontal Position"
                )
                position_y = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.7, 
                    label="Vertical Position"
                )
                scale = gr.Slider(
                    minimum=0.1, maximum=2.0, value=1.0, 
                    label="Person Scale"
                )
            
            with gr.Accordion("Anaglyph Settings", open=True):
                anaglyph_method = gr.Radio(
                    choices=["true", "optimized", "gray"], 
                    label="Anaglyph Method", 
                    value="true"
                )
                brightness = gr.Slider(
                    minimum=0.7, maximum=1.3, value=1.0, 
                    label="Brightness"
                )
                contrast = gr.Slider(
                    minimum=0.7, maximum=1.3, value=1.0, 
                    label="Contrast"
                )
            
            process_btn = gr.Button("Process Image")
        
        with gr.Column(scale=2):
            with gr.Tab("Segmented Person"):
                segmented_output = gr.Image(label="Segmented Person")
                
            with gr.Tab("Stereo Pair"):
                with gr.Row():
                    left_output = gr.Image(label="Left Image")
                    right_output = gr.Image(label="Right Image")
            
            with gr.Tab("Anaglyph Result"):
                anaglyph_output = gr.Image(label="Anaglyph (View with Red-Cyan glasses)")
    
    # Connect the button to the processing function
    process_btn.click(
        fn=process_image,
        inputs=[
            input_image, depth, position_x, position_y, scale,
            anaglyph_method, brightness, contrast
        ],
        outputs=[segmented_output, left_output, right_output, anaglyph_output]
    )
    
    # Add examples
    gr.Examples(
        examples=[
            [None, "close", 0.5, 0.7, 1.0, "true", 1.0, 1.0],
            [None, "medium", 0.3, 0.7, 1.2, "optimized", 1.1, 1.0],
            [None, "far", 0.7, 0.6, 0.8, "gray", 1.0, 1.1]
        ],
        inputs=[
            input_image, depth, position_x, position_y, scale,
            anaglyph_method, brightness, contrast
        ]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()