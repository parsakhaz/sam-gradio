import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor
import cv2
import colorsys
import random

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

def generate_color_pair():
    # Generate a random hue
    hue = random.random()
    
    # Create a darker version (lower brightness/value)
    dark_rgb = [int(255 * x) for x in colorsys.hsv_to_rgb(hue, 0.8, 0.7)]
    
    # Create a lighter version (higher brightness/value, lower saturation)
    light_rgb = [int(255 * x) for x in colorsys.hsv_to_rgb(hue, 0.6, 0.9)]
    
    return dark_rgb, light_rgb

def create_mask_overlay(image, mask):
    # Convert binary mask to uint8
    mask_uint8 = (mask > 0).astype(np.uint8)
    
    # Dilation to fill gaps
    kernel = np.ones((5, 5), np.uint8)  # You can adjust the kernel size
    mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    
    # Find contours of the dilated mask
    contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Generate random color pair for this segmentation
    dark_color, light_color = generate_color_pair()
    
    # Create a transparent overlay for the mask
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask_dilated > 0] = [*light_color, 90]  # Light color with 35% opacity
    
    # Create a separate layer for the outline
    outline = np.zeros((*mask.shape, 4), dtype=np.uint8)
    cv2.drawContours(outline, contours, -1, (*dark_color, 255), 2)  # Dark color outline
    
    # Convert to PIL images
    mask_overlay = Image.fromarray(overlay, 'RGBA')
    outline_overlay = Image.fromarray(outline, 'RGBA')
    
    # Composite the layers
    result = image.convert('RGBA')
    result.paste(mask_overlay, (0, 0), mask_overlay)
    result.paste(outline_overlay, (0, 0), outline_overlay)
    
    return result

def process_image(image, x, y):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    inputs = sam_processor(
        image,
        input_points=[[[x, y]]],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    mask = sam_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )[0][0][0].numpy()
    
    # Use the visualization function
    result = create_mask_overlay(image, mask)
    return result.convert('RGB')  # Convert back to RGB for web display

def segment_image(image, x_coord, y_coord):
    if image is None:
        raise gr.Error("Please upload an image first")
    
    try:
        # Convert coordinates to integers
        x = int(x_coord)
        y = int(y_coord)
        
        # Process with SAM model
        result = process_image(image, x, y)
        
        return result
    except ValueError:
        raise gr.Error("Please enter valid numbers for coordinates")
    except Exception as e:
        raise gr.Error(f"Error processing image: {str(e)}")

# Create the Gradio interface
with gr.Blocks(title="SAM Demo") as demo:
    gr.Markdown("# Segment Anything Demo")
    gr.Markdown("Upload an image and enter coordinates to segment an object at that location.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input Image",
                type="pil",
                height=512
            )
            with gr.Row():
                x_input = gr.Number(label="X Coordinate", value=0)
                y_input = gr.Number(label="Y Coordinate", value=0)
            segment_btn = gr.Button("Segment Object")
        
        with gr.Column():
            output_image = gr.Image(
                label="SAM Output",
                height=512
            )
    
    segment_btn.click(
        fn=segment_image,
        inputs=[input_image, x_input, y_input],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Standard Gradio port
        share=True             # Create a public link
    ) 