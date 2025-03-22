"""
3D Image Composer - Improved Layout with Combined Upload/Selection Section
"""

import os
import cv2
import numpy as np
import gradio as gr
from PIL import Image
import tempfile
import glob

# Import our utility modules
from utils.segmentation import load_segmentation_model, segment_person
from utils.stereo_processing import load_stereo_pair, insert_person_with_depth
from utils.anaglyph import create_anaglyph

# Sample data paths - matching your directory structure
SAMPLE_PERSONS_DIR = "data/raw_person"
SAMPLE_STEREO_DIR = "data/stereoscopic_images"

# Load the segmentation model once at startup
model, device = load_segmentation_model()

# Get sample image paths
def get_sample_images():
    person_samples = []
    stereo_samples = []
    
    if os.path.exists(SAMPLE_PERSONS_DIR):
        person_samples = glob.glob(os.path.join(SAMPLE_PERSONS_DIR, "*.jpg")) + \
                         glob.glob(os.path.join(SAMPLE_PERSONS_DIR, "*.jpeg")) + \
                         glob.glob(os.path.join(SAMPLE_PERSONS_DIR, "*.png"))
    
    if os.path.exists(SAMPLE_STEREO_DIR):
        stereo_samples = glob.glob(os.path.join(SAMPLE_STEREO_DIR, "*.jpg")) + \
                         glob.glob(os.path.join(SAMPLE_STEREO_DIR, "*.jpeg")) + \
                         glob.glob(os.path.join(SAMPLE_STEREO_DIR, "*.png"))
    
    return person_samples, stereo_samples

# Process images function
def process_images(person_image, stereo_image, depth_level):
    # Make sure we have both inputs
    if person_image is None:
        return None, "Please select a person image."
    if stereo_image is None:
        return None, "Please select a stereoscopic image."
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    
    # Handle person image
    person_path = os.path.join(temp_dir, "person.jpg")
    if isinstance(person_image, str) and os.path.exists(person_image):
        person_path = person_image
    else:
        try:
            if isinstance(person_image, np.ndarray):
                cv2.imwrite(person_path, person_image)
            else:
                Image.fromarray(np.array(person_image)).save(person_path)
        except Exception as e:
            return None, f"Error processing person image: {str(e)}"
    
    # Handle stereo image
    stereo_path = os.path.join(temp_dir, "stereo.jpg")
    if isinstance(stereo_image, str) and os.path.exists(stereo_image):
        stereo_path = stereo_image
    else:
        try:
            if isinstance(stereo_image, np.ndarray):
                cv2.imwrite(stereo_path, stereo_image)
            else:
                Image.fromarray(np.array(stereo_image)).save(stereo_path)
        except Exception as e:
            return None, f"Error processing stereo image: {str(e)}"
    
    try:
        # Step 1: Segment the person
        segmented_person = segment_person(person_path, model, device)
        
        # Save the segmented person for display
        segmented_path = os.path.join(temp_dir, "segmented_person.png")
        cv2.imwrite(segmented_path, cv2.cvtColor(segmented_person, cv2.COLOR_RGBA2BGRA))
        
        # Step 2: Load and process the stereoscopic image
        left_bg, right_bg = load_stereo_pair(stereo_path)
        
        # Step 3: Insert the person into both images with the specified depth
        left_composite, right_composite = insert_person_with_depth(
            left_bg, right_bg, segmented_person, depth_level
        )
        
        # Step 4: Create the anaglyph image
        anaglyph = create_anaglyph(left_composite, right_composite)
        
        # Save the anaglyph for display
        anaglyph_path = os.path.join(temp_dir, "anaglyph.jpg")
        cv2.imwrite(anaglyph_path, anaglyph)
        
        return segmented_path, anaglyph_path
    
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create the Gradio interface
def create_interface():
    person_samples, stereo_samples = get_sample_images()
    
    with gr.Blocks(title="3D Image Composer") as app:
        gr.Markdown("# 3D Image Composer")
        
        # State to track selections
        selected_person = gr.State(None)
        selected_stereo = gr.State(None)
        
        # Combined upload and selection display section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Person Image")
                # Display the selected or uploaded person image
                selected_person_img = gr.Image(label="Currently Selected/Uploaded Person", height=200)
                
                # Upload option
                custom_person = gr.Image(type="filepath", label="Upload Person Image")
                
                # Function to handle person image changes
                def update_person_display(uploaded, selected):
                    if uploaded is not None:
                        return uploaded, uploaded
                    else:
                        return selected, selected
                
                custom_person.change(
                    fn=update_person_display,
                    inputs=[custom_person, selected_person],
                    outputs=[selected_person_img, selected_person]
                )
            
            with gr.Column():
                gr.Markdown("### Stereoscopic Background")
                # Display the selected or uploaded stereo image
                selected_stereo_img = gr.Image(label="Currently Selected/Uploaded Background", height=200)
                
                # Upload option
                custom_stereo = gr.Image(type="filepath", label="Upload Stereoscopic Background")
                
                # Function to handle stereo image changes
                def update_stereo_display(uploaded, selected):
                    if uploaded is not None:
                        return uploaded, uploaded
                    else:
                        return selected, selected
                
                custom_stereo.change(
                    fn=update_stereo_display,
                    inputs=[custom_stereo, selected_stereo],
                    outputs=[selected_stereo_img, selected_stereo]
                )
        
        gr.Markdown("### Choose from Sample Images:")
        
        # Person gallery section
        gr.Markdown("#### Person Images:")
        
        # Create simple row for person images with click handlers
        with gr.Row():
            person_img_components = []
            
            # Display up to 4 person images per row for better visibility
            for i, img_path in enumerate(person_samples[:8]):
                with gr.Column(scale=1, min_width=100):
                    img = gr.Image(value=img_path, show_label=False, height=150)
                    person_img_components.append(img)
                    
                    # Add a select button under each image
                    select_btn = gr.Button(f"Select")
                    
                    # Set up click handler to update selection and display
                    select_btn.click(
                        fn=lambda path=img_path: (path, path),
                        outputs=[selected_person_img, selected_person]
                    )
        
        # Stereo gallery section
        gr.Markdown("#### Stereoscopic Background Images:")
        
        # Simple row for stereo images with click handlers
        with gr.Row():
            stereo_img_components = []
            
            # Display up to 4 stereo images per row
            for i, img_path in enumerate(stereo_samples[:8]):
                with gr.Column(scale=1, min_width=150):
                    img = gr.Image(value=img_path, show_label=False, height=150)
                    stereo_img_components.append(img)
                    
                    # Add a select button under each image
                    select_btn = gr.Button(f"Select")
                    
                    # Set up click handler to update selection and display
                    select_btn.click(
                        fn=lambda path=img_path: (path, path),
                        outputs=[selected_stereo_img, selected_stereo]
                    )
        
        # Depth selection
        with gr.Row():
            depth_selector = gr.Radio(
                choices=["close", "medium", "far"],
                value="medium",
                label="Depth Level",
                info="Choose how far the person appears in the 3D scene"
            )
            
            # Create button
            process_btn = gr.Button("Create 3D Image", variant="primary", size="lg")
        
        # Outputs
        with gr.Row():
            segmented_output = gr.Image(label="Segmented Person")
            anaglyph_output = gr.Image(label="Anaglyph Image (View with Red-Cyan Glasses)")
        
        # Processing function
        def process_with_choices(person_path, stereo_path, depth):
            if person_path is None:
                return None, "Please select or upload a person image."
            if stereo_path is None:
                return None, "Please select or upload a stereoscopic background image."
            
            return process_images(person_path, stereo_path, depth)
        
        # Connect the button
        process_btn.click(
            fn=process_with_choices,
            inputs=[
                selected_person, 
                selected_stereo,
                depth_selector
            ],
            outputs=[segmented_output, anaglyph_output]
        )
        
        gr.Markdown("""
        ## How to use:
        1. Either upload your own images or click "Select" under a sample image to choose it
        2. Your selected images will appear in the top section
        3. Choose the depth level: close, medium, or far
        4. Click "Create 3D Image"
        5. View the result with red-cyan 3D glasses
        """)
    
    return app

# Launch the app
if __name__ == "__main__":
    app = create_interface()
    app.launch()