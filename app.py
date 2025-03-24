"""
3D Image Composer - Simplified Single-Page Layout
Compatible with older Gradio versions
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

# Sample data paths
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
        return None, None, "Please select a person image."
    if stereo_image is None:
        return None, None, "Please select a stereoscopic image."
    
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
            return None, None, f"Error processing person image: {str(e)}"
    
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
            return None, None, f"Error processing stereo image: {str(e)}"
    
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
        
        return segmented_path, anaglyph_path, None
    
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# Create the Gradio interface with simplified layout
def create_interface():
    person_samples, stereo_samples = get_sample_images()
    
    with gr.Blocks(title="3D Image Composer") as app:
        gr.Markdown(
            """
            # 3D Image Composer
            *Create amazing 3D images by placing people into stereoscopic backgrounds. View the results with red-cyan 3D glasses!*
            """
        )
        
        # State to track selections
        selected_person = gr.State(None)
        selected_stereo = gr.State(None)
        
        # Main sections in a single-page layout
        with gr.Row():
            # Left side: Image selection panel
            with gr.Column(scale=1):
                gr.Markdown("## 📸 Select Your Images")
                
                # Person image section
                with gr.Group():
                    gr.Markdown("### 🧍 Person Image")
                    
                    # Person image display
                    selected_person_img = gr.Image(
                        label="Selected Person",
                        elem_id="selected_person",
                        height=250
                    )
                    
                    # Upload button
                    custom_person = gr.Image(
                        type="filepath", 
                        label="Upload a Person Image", 
                        elem_id="person_upload"
                    )
                    
                    # Sample person images
                    gr.Markdown("#### Or choose from samples:")
                    
                    # Use a single row with scrolling for better space usage
                    with gr.Row():
                        # Show just the first few samples to save space
                        for i, path in enumerate(person_samples[:4]):
                            with gr.Column(scale=1, min_width=100):
                                sample_img = gr.Image(value=path, show_label=False, height=100)
                                sample_btn = gr.Button("Select")
                                
                                # Create a function to handle selection
                                def make_select_fn(path):
                                    return lambda: (path, path)
                                
                                sample_btn.click(
                                    fn=make_select_fn(path),
                                    outputs=[selected_person_img, selected_person]
                                )
                
                # Stereo image section
                with gr.Group():
                    gr.Markdown("### 🌄 Stereoscopic Background")
                    
                    # Stereo image display
                    selected_stereo_img = gr.Image(
                        label="Selected Background",
                        elem_id="selected_stereo",
                        height=250
                    )
                    
                    # Upload button
                    custom_stereo = gr.Image(
                        type="filepath", 
                        label="Upload a Stereoscopic Background", 
                        elem_id="stereo_upload"
                    )
                    
                    # Sample stereo images
                    gr.Markdown("#### Or choose from samples:")
                    
                    # Use a single row with scrolling for better space usage
                    with gr.Row():
                        # Show just the first few samples to save space
                        for i, path in enumerate(stereo_samples[:4]):
                            with gr.Column(scale=1, min_width=100):
                                sample_img = gr.Image(value=path, show_label=False, height=100)
                                sample_btn = gr.Button("Select")
                                
                                def make_select_fn(path):
                                    return lambda: (path, path)
                                
                                sample_btn.click(
                                    fn=make_select_fn(path),
                                    outputs=[selected_stereo_img, selected_stereo]
                                )
                
                # Depth settings section
                with gr.Group():
                    gr.Markdown("### 🎛️ Depth Settings")
                    depth_selector = gr.Radio(
                        choices=["close", "medium", "far"],
                        value="medium",
                        label="Person's Position in 3D Space",
                        info="How far the person appears in the 3D scene"
                    )
                    
                    gr.Markdown("""
                    * **Close**: Person appears very close to viewer
                    * **Medium**: Person appears at a moderate distance
                    * **Far**: Person appears in the background
                    """)
                
                # Create button
                process_btn = gr.Button("🔮 Generate 3D Image", variant="primary", size="lg")
            
            # Right side: Results panel
            with gr.Column(scale=1):
                gr.Markdown("## 🖼️ Results")
                
                # Status message
                status_msg = gr.Markdown("### Select images and click Generate to create your 3D image")
                
                # Segmented person output
                gr.Markdown("### Segmented Person")
                segmented_output = gr.Image(label="", height=250)
                
                # Final anaglyph output
                gr.Markdown("### 🥽 3D Anaglyph Image")
                anaglyph_output = gr.Image(label="View with Red-Cyan 3D Glasses", height=400)
                
                # Tips for viewing
                with gr.Accordion("📋 Tips for viewing", open=False):
                    gr.Markdown("""
                    * For the best 3D effect, view the image with red-cyan anaglyph glasses
                    * The left eye should look through the red filter, and the right eye through the cyan filter
                    * Adjust your viewing distance for comfort - typically arm's length works well
                    * To download the 3D image, right-click on it and select "Save image as..."
                    """)
        
        # Help section at the bottom
        with gr.Accordion("ℹ️ About 3D Image Composer", open=False):
            gr.Markdown("""
            ## How to use this app:
            
            1. Select or upload a person image on the left panel
            2. Select or upload a stereoscopic background image
            3. Choose how far the person should appear in the 3D scene
            4. Click "Generate 3D Image"
            5. View the resulting 3D anaglyph with red-cyan glasses
            
            ## What is a stereoscopic image?
            
            A stereoscopic image contains two slightly different views of the same scene, similar to what your left and right eyes see. The app uses side-by-side stereoscopic images where the left half is for the left eye and the right half for the right eye.
            
            ## About 3D glasses
            
            Red-cyan anaglyph glasses work by filtering different colors to each eye. The red filter blocks cyan light, and the cyan filter blocks red light. This allows each eye to see a different image, creating the illusion of depth.
            """)
        
        # Function handlers
        
        # Handle custom person upload
        custom_person.change(
            fn=lambda x: (x, x),
            inputs=[custom_person],
            outputs=[selected_person_img, selected_person]
        )
        
        # Handle custom stereo upload
        custom_stereo.change(
            fn=lambda x: (x, x),
            inputs=[custom_stereo],
            outputs=[selected_stereo_img, selected_stereo]
        )
        
        # Processing function with status updates
        def process_with_status(person_path, stereo_path, depth):
            # Input validation
            if person_path is None:
                return None, None, "⚠️ Please select a person image first."
            if stereo_path is None:
                return None, None, "⚠️ Please select a stereoscopic background image first."
            
            # Process images
            segmented, anaglyph, error = process_images(person_path, stereo_path, depth)
            
            if error:
                return None, None, f"❌ {error}"
            else:
                return segmented, anaglyph, "✅ 3D image created successfully! View with red-cyan glasses."
        
        # Connect the processing button
        process_btn.click(
            fn=process_with_status,
            inputs=[selected_person, selected_stereo, depth_selector],
            outputs=[segmented_output, anaglyph_output, status_msg]
        )
    
    return app

# Launch the app
if __name__ == "__main__":
    app = create_interface()
    app.launch()