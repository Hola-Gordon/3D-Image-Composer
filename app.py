"""
3D Image Composer - Complete app with position controls and background adaptation
Features:
- Person segmentation using DeepLabV3
- Stereoscopic image processing
- Anaglyph creation for 3D viewing
- Advanced positioning and depth controls
- Background color adaptation
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
from utils.stereo_processing import load_stereo_pair, insert_person_with_depth_position
from utils.anaglyph import create_anaglyph

# Define paths to sample data directories
SAMPLE_PERSONS_DIR = "data/raw_person"
SAMPLE_STEREO_DIR = "data/stereoscopic_images"

# Load the segmentation model once at startup for efficiency
model, device = load_segmentation_model()

def get_sample_images():
    """
    Get lists of sample images from the data directories.
    
    Returns:
        Tuple of (person_samples, stereo_samples) with file paths
    """
    person_samples = []
    stereo_samples = []
    
    # Check if directories exist and collect image paths
    if os.path.exists(SAMPLE_PERSONS_DIR):
        person_samples = glob.glob(os.path.join(SAMPLE_PERSONS_DIR, "*.jpg")) + \
                         glob.glob(os.path.join(SAMPLE_PERSONS_DIR, "*.jpeg")) + \
                         glob.glob(os.path.join(SAMPLE_PERSONS_DIR, "*.png"))
    
    if os.path.exists(SAMPLE_STEREO_DIR):
        stereo_samples = glob.glob(os.path.join(SAMPLE_STEREO_DIR, "*.jpg")) + \
                         glob.glob(os.path.join(SAMPLE_STEREO_DIR, "*.jpeg")) + \
                         glob.glob(os.path.join(SAMPLE_STEREO_DIR, "*.png"))
    
    return person_samples, stereo_samples

def process_images(person_image, stereo_image, depth_level, x_position, y_position, scale_factor, adapt_colors):
    """
    Main processing function to create a 3D anaglyph from person and background images.
    
    Args:
        person_image: Path or array of the person image
        stereo_image: Path or array of the stereoscopic background
        depth_level: Depth setting ('close', 'medium', 'far')
        x_position: Horizontal position percentage (0-100)
        y_position: Vertical position percentage (0-100)
        scale_factor: Size scaling factor
        adapt_colors: Whether to adapt person colors to match background
        
    Returns:
        Tuple of (segmented_path, anaglyph_path, error_message)
    """
    # Validate inputs
    if person_image is None:
        return None, None, "Please select a person image."
    if stereo_image is None:
        return None, None, "Please select a stereoscopic image."
    
    # Create temporary directory for processing files
    temp_dir = tempfile.mkdtemp()
    
    # Handle person image - convert to file path if needed
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
    
    # Handle stereo image - convert to file path if needed
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
        # Step 1: Segment the person from the background
        segmented_person = segment_person(person_path, model, device)
        
        # Save the segmented person image for display
        segmented_path = os.path.join(temp_dir, "segmented_person.png")
        cv2.imwrite(segmented_path, cv2.cvtColor(segmented_person, cv2.COLOR_RGBA2BGRA))
        
        # Step 2: Load and process the stereoscopic image
        left_bg, right_bg = load_stereo_pair(stereo_path)
        
        # Step 3: Insert the person into both images with specified parameters
        left_composite, right_composite = insert_person_with_depth_position(
            left_bg, right_bg, segmented_person, depth_level, 
            x_position, y_position, scale_factor, adapt_colors
        )
        
        # Step 4: Create the anaglyph image by combining left and right views
        anaglyph = create_anaglyph(left_composite, right_composite)
        
        # Save the anaglyph for display
        anaglyph_path = os.path.join(temp_dir, "anaglyph.jpg")
        cv2.imwrite(anaglyph_path, anaglyph)
        
        return segmented_path, anaglyph_path, None
    
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def create_interface():
    """
    Create the Gradio interface for the 3D Image Composer app.
    
    Returns:
        Gradio Blocks interface
    """
    # Get sample images
    person_samples, stereo_samples = get_sample_images()
    
    # Create the Gradio interface
    with gr.Blocks(title="3D Image Composer") as app:
        # Header section
        gr.Markdown(
            """
            # 3D Image Composer
            *Create amazing 3D images by placing people into stereoscopic backgrounds. View the results with red-cyan 3D glasses!*
            """
        )
        
        # State variables to track selections
        selected_person = gr.State(None)
        selected_stereo = gr.State(None)
        
        # Image selection row - Person and Background side by side
        gr.Markdown("## 📸 Select Your Images")
        
        with gr.Row():
            # Person image section
            with gr.Column():
                gr.Markdown("### 🧍 Person Image")
                
                # Person image display
                selected_person_img = gr.Image(
                    label="Selected Person",
                    elem_id="selected_person",
                    height=200
                )
                
                # Upload button
                custom_person = gr.Image(
                    type="filepath", 
                    label="Upload a Person Image", 
                    elem_id="person_upload"
                )
                
                # Sample person images
                gr.Markdown("#### Or choose from samples:")
                
                # Show sample images in a compact grid
                with gr.Row():
                    for i, path in enumerate(person_samples[:4]):
                        with gr.Column(scale=1, min_width=80):
                            sample_img = gr.Image(value=path, show_label=False, height=80)
                            sample_btn = gr.Button("Select")
                            
                            # Create a function to handle selection
                            def make_select_fn(path):
                                return lambda: (path, path)
                            
                            sample_btn.click(
                                fn=make_select_fn(path),
                                outputs=[selected_person_img, selected_person]
                            )
            
            # Stereo image section
            with gr.Column():
                gr.Markdown("### 🌄 Stereoscopic Background")
                
                # Stereo image display
                selected_stereo_img = gr.Image(
                    label="Selected Background",
                    elem_id="selected_stereo",
                    height=200
                )
                
                # Upload button
                custom_stereo = gr.Image(
                    type="filepath", 
                    label="Upload a Stereoscopic Background", 
                    elem_id="stereo_upload"
                )
                
                # Sample stereo images
                gr.Markdown("#### Or choose from samples:")
                
                # Show sample images in a compact grid
                with gr.Row():
                    for i, path in enumerate(stereo_samples[:4]):
                        with gr.Column(scale=1, min_width=80):
                            sample_img = gr.Image(value=path, show_label=False, height=80)
                            sample_btn = gr.Button("Select")
                            
                            def make_select_fn(path):
                                return lambda: (path, path)
                            
                            sample_btn.click(
                                fn=make_select_fn(path),
                                outputs=[selected_stereo_img, selected_stereo]
                            )
        
        # Controls and Results section
        with gr.Row():
            # Controls column
            with gr.Column(scale=1):
                gr.Markdown("## 🎛️ Controls")
                
                # Position and Depth Controls
                with gr.Group():
                    gr.Markdown("### Position and Depth Settings")
                    
                    # Depth selector
                    depth_selector = gr.Radio(
                        choices=["close", "medium", "far"],
                        value="medium",
                        label="Person's Depth in 3D Space",
                        info="How far the person appears in the 3D scene"
                    )
                    
                    # Horizontal position slider
                    x_position = gr.Slider(
                        minimum=0, 
                        maximum=100, 
                        value=50, 
                        step=1, 
                        label="Horizontal Position (%)",
                        info="Move the person left to right"
                    )
                    
                    # Vertical position slider
                    y_position = gr.Slider(
                        minimum=0, 
                        maximum=100, 
                        value=75, 
                        step=1, 
                        label="Vertical Position (%)",
                        info="Move the person up and down"
                    )
                    
                    # Scale slider
                    scale_slider = gr.Slider(
                        minimum=0.5, 
                        maximum=3.0, 
                        value=1, 
                        step=0.01, 
                        label="Size Scale",
                        info="Adjust the size of the person"
                    )
                    
                    # Color adaptation checkbox - Advanced feature
                    adapt_colors = gr.Checkbox(
                        value=True,
                        label="Adapt Colors",
                        info="Match person's lighting and color to the background"
                    )
                
                # Create button
                process_btn = gr.Button("🔮 Generate 3D Image", variant="primary", size="lg")
                
                # Segmented person output
                gr.Markdown("### Segmented Person")
                segmented_output = gr.Image(label="", height=200)
            
            # Results column
            with gr.Column(scale=1):
                gr.Markdown("## 🖼️ 3D Result")
                
                # Status message
                status_msg = gr.Markdown("### Select images and click Generate to create your 3D image")
                
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
            
            1. Select or upload a person image and stereoscopic background 
            2. Adjust the position, depth, and size controls
            3. Toggle "Adapt Colors" to match lighting between person and background
            4. Click "Generate 3D Image"
            5. View the 3D anaglyph with red-cyan glasses
            
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
        def process_with_status(person_path, stereo_path, depth, x_pos, y_pos, scale, adapt_colors):
            """
            Process images and provide status updates to the user.
            
            Args:
                person_path: Path to the person image
                stereo_path: Path to the stereo image
                depth: Depth level selection
                x_pos: Horizontal position
                y_pos: Vertical position
                scale: Size scale
                adapt_colors: Whether to apply color adaptation
                
            Returns:
                Tuple of (segmented_output, anaglyph_output, status_message)
            """
            # Input validation
            if person_path is None:
                return None, None, "⚠️ Please select a person image first."
            if stereo_path is None:
                return None, None, "⚠️ Please select a stereoscopic background image first."
            
            # Process images with all parameters
            segmented, anaglyph, error = process_images(
                person_path, stereo_path, depth, x_pos, y_pos, scale, adapt_colors
            )
            
            # Handle results or errors
            if error:
                return None, None, f"❌ {error}"
            else:
                return segmented, anaglyph, "✅ 3D image created successfully! View with red-cyan glasses."
        
        # Connect the processing button to the function
        process_btn.click(
            fn=process_with_status,
            inputs=[
                selected_person, 
                selected_stereo, 
                depth_selector, 
                x_position, 
                y_position, 
                scale_slider,
                adapt_colors
            ],
            outputs=[segmented_output, anaglyph_output, status_msg]
        )
    
    return app

# Launch the app when this script is run directly
if __name__ == "__main__":
    app = create_interface()
    app.launch()