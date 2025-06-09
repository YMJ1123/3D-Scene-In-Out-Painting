import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
os.chdir("../")
import cv2
import gradio as gr
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import tempfile
# from omegaconf import OmegaConf
# from sam_segment import predict_masks_with_sam
from stable_diffusion_inpaint import replace_img_with_sd
from lama_inpaint import inpaint_img_with_lama, build_lama_model, inpaint_img_with_builded_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import argparse

def setup_args(parser):
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str,
        default="pretrained_models/big-lama",
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--sam_ckpt", type=str,
        default="./pretrained_models/sam_vit_h_4b8939.pth",
        help="The path to the SAM checkpoint to use for mask generation.",
    )
def mkstemp(suffix, dir=None):
    fd, path = tempfile.mkstemp(suffix=f"{suffix}", dir=dir)
    os.close(fd)
    return Path(path)


def get_sam_feat(img):
    model['sam'].set_image(img)
    features = model['sam'].features 
    orig_h = model['sam'].orig_h 
    orig_w = model['sam'].orig_w 
    input_h = model['sam'].input_h 
    input_w = model['sam'].input_w 
    model['sam'].reset_image()
    return features, orig_h, orig_w, input_h, input_w

def get_replace_img_with_sd(image, mask, image_resolution, text_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(mask.shape)==3:
        mask = mask[:,:,0]
    np_image = np.array(image, dtype=np.uint8)
    H, W, C = np_image.shape
    np_image = HWC3(np_image)
    np_image = resize_image(np_image, image_resolution)

    img_replaced = replace_img_with_sd(np_image, mask, text_prompt, device=device)
    img_replaced = img_replaced.astype(np.uint8)
    return img_replaced

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def resize_points(clicked_points, original_shape, resolution):
    original_height, original_width, _ = original_shape
    original_height = float(original_height)
    original_width = float(original_width)
    
    scale_factor = float(resolution) / min(original_height, original_width)
    resized_points = []
    
    for point in clicked_points:
        x, y, lab = point
        resized_x = int(round(x * scale_factor))
        resized_y = int(round(y * scale_factor))
        resized_point = (resized_x, resized_y, lab)
        resized_points.append(resized_point)
    
    return resized_points

def get_click_mask(clicked_points, features, orig_h, orig_w, input_h, input_w):
    # model['sam'].set_image(image)
    model['sam'].is_image_set = True
    model['sam'].features = features
    model['sam'].orig_h = orig_h
    model['sam'].orig_w = orig_w
    model['sam'].input_h = input_h
    model['sam'].input_w = input_w
    
    points, labels = zip(*[(point[:2], point[2])
                            for point in clicked_points])

    input_point = np.array(points)
    input_label = np.array(labels)

    masks, _, _ = model['sam'].predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size.value) for mask in masks]
    else:
        masks = [mask for mask in masks]

    return masks

def process_image_click(original_image, point_prompt, clicked_points, image_resolution, features, orig_h, orig_w, input_h, input_w, evt: gr.SelectData):
    clicked_coords = evt.index
    x, y = clicked_coords
    label = point_prompt
    lab = 1 if label == "Foreground Point" else 0
    clicked_points.append((x, y, lab))

    input_image = np.array(original_image, dtype=np.uint8)
    H_orig, W_orig, C = input_image.shape # Renamed to H_orig, W_orig for clarity
    # The input_image here is the state `origin_image` or `new_object_image_input` which are full resolution.
    # SAM features are derived from a resized version of this image using image_resolution.
    # So points need to be scaled to that resized version for SAM.

    # Create a temporary resized version of input_image for SAM if points are in original image coords
    # Or, ensure points are already scaled for the SAM model's input resolution.
    # The current resize_points function scales points from original image to the SAM input resolution.
    resized_points = resize_points(
        clicked_points, input_image.shape, image_resolution
    )
    mask_click_np = get_click_mask(resized_points, features, orig_h, orig_w, input_h, input_w)

    mask_click_np = np.transpose(mask_click_np, (1, 2, 0)) * 255.0
    mask_image_sam_size = HWC3(mask_click_np.astype(np.uint8))
    
    # Resize mask back to original image dimensions (H_orig, W_orig)
    mask_image = cv2.resize(
        mask_image_sam_size, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR
    )

    # Draw circles on the original full-resolution image copy
    edited_image_display = input_image.copy()
    for pt_x, pt_y, pt_lab in clicked_points:
        color = (255, 0, 0) if pt_lab == 1 else (0, 0, 255)
        edited_image_display = cv2.circle(edited_image_display, (pt_x, pt_y), 20, color, -1)

    opacity_mask = 0.75
    opacity_edited = 1.0

    overlay_image = cv2.addWeighted(
        edited_image_display,
        opacity_edited,
        (mask_image * np.array([0/255, 255/255, 0/255])).astype(np.uint8),
        opacity_mask,
        0,
    )

    return (
        overlay_image, # This is original_image with points and mask overlay
        clicked_points,
        mask_image,  # Mask at original image resolution
        mask_image   # Mask at original image resolution (for state)
    )

def image_upload(image, image_resolution):
    if image is not None:
        np_image_orig = np.array(image, dtype=np.uint8) # Original uploaded image
        np_image_orig = HWC3(np_image_orig)
        # For SAM, resize to image_resolution
        np_image_sam_input = resize_image(np_image_orig.copy(), image_resolution) 
        features, orig_h, orig_w, input_h, input_w = get_sam_feat(np_image_sam_input)
        # Return the original image for state, and SAM features
        return image, features, orig_h, orig_w, input_h, input_w 
    else:
        return None, None, None, None, None, None

def get_inpainted_img(image, mask, image_resolution):
    lama_config = args.lama_config
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure image and mask are numpy arrays
    image_np = np.array(image, dtype=np.uint8)
    mask_np = np.array(mask, dtype=np.uint8)

    if mask_np is not None and len(mask_np.shape) == 3:
        mask_np = mask_np[:, :, 0] # Ensure single channel for lama
    elif mask_np is None or mask_np.size == 0:
        return image_np # Return original image if mask is invalid

    # Ensure mask is binary 0 or 255
    if mask_np.max() == 1: # If it's boolean 0/1
        mask_np = (mask_np * 255).astype(np.uint8)
    else: # Ensure it's uint8 type, threshold just in case
        mask_np = mask_np.astype(np.uint8)
    _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    # Lama expects BGR images, but Gradio usually handles RGB. Let's assume input `image` is RGB.
    # The `inpaint_img_with_builded_lama` internally handles BGR conversion if needed by the model and returns RGB.
    img_inpainted = inpaint_img_with_builded_lama(
        model['lama'], image_np, mask_np, lama_config, device=device)
    return img_inpainted

# Helper function to prepare a binary mask
def _prepare_binary_mask(mask_input, target_h=None, target_w=None):
    if mask_input is None or not hasattr(mask_input, 'size') or mask_input.size == 0:
        if target_h is not None and target_w is not None:
            return np.zeros((target_h, target_w), dtype=np.uint8)
        return None

    mask = np.array(mask_input) # Ensure it's a numpy array

    if mask.ndim == 3:
        mask = mask[:, :, 0] # Take first channel if 3-channel

    # Convert to 0-255 uint8 if it's boolean or 0-1 float
    if mask.dtype == bool or \
       (np.issubdtype(mask.dtype, np.floating) and mask.max() <= 1.0 and mask.min() >= 0.0 and mask.size > 0):
        mask = (mask * 255).astype(np.uint8)
    elif mask.size > 0: # Ensure it's uint8 if not already boolean/float01
        mask = mask.astype(np.uint8)
    else: # Empty mask after processing, treat as None scenario from start
        if target_h is not None and target_w is not None:
            return np.zeros((target_h, target_w), dtype=np.uint8)
        return None

    # Resize if target dimensions are provided
    if target_h is not None and target_w is not None:
        current_h, current_w = mask.shape[:2]
        if current_h != target_h or current_w != target_w:
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    # Binarize to 0 and 255
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

def composite_object_on_inpainted(inpainted_bg_input, new_obj_img_full_input, new_obj_mask_input, original_target_mask_input):
    if inpainted_bg_input is None or new_obj_img_full_input is None:
        return inpainted_bg_input if inpainted_bg_input is not None else np.zeros((256, 256, 3), dtype=np.uint8)

    inpainted_bg = HWC3(np.array(inpainted_bg_input, dtype=np.uint8))
    new_obj_img_full = HWC3(np.array(new_obj_img_full_input, dtype=np.uint8))
    
    H_bg, W_bg = inpainted_bg.shape[:2]

    new_obj_mask_1ch_bin = _prepare_binary_mask(new_obj_mask_input)
    if new_obj_mask_1ch_bin is None or new_obj_mask_1ch_bin.sum() == 0: # if mask is empty or all zeros
        return inpainted_bg # No object selected from new image

    # Align original_target_mask with inpainted_bg dimensions and binarize
    original_target_mask_1ch_bin = _prepare_binary_mask(original_target_mask_input, target_h=H_bg, target_w=W_bg)
    if original_target_mask_1ch_bin is None or original_target_mask_1ch_bin.sum() == 0: # if mask is empty or all zeros
        return inpainted_bg # No target area defined in the background

    contours_new_obj, _ = cv2.findContours(new_obj_mask_1ch_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_new_obj:
        return inpainted_bg # No object contour found
    # Combine all contours to get a bounding box for the entire selected object area
    all_points_new_obj = np.concatenate(contours_new_obj)
    nx, ny, nw, nh = cv2.boundingRect(all_points_new_obj)
    if nw == 0 or nh == 0: return inpainted_bg

    cropped_object_unmasked = new_obj_img_full[ny:ny + nh, nx:nx + nw]
    cropped_object_mask_1ch = new_obj_mask_1ch_bin[ny:ny + nh, nx:nx + nw]

    contours_target, _ = cv2.findContours(original_target_mask_1ch_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_target:
        return inpainted_bg # No target area contour found
    all_points_target = np.concatenate(contours_target)
    tx, ty, tw, th = cv2.boundingRect(all_points_target)
    if tw == 0 or th == 0: return inpainted_bg

    resized_object_unmasked = cv2.resize(cropped_object_unmasked, (tw, th))
    # Use INTER_NEAREST for mask to maintain binary nature as much as possible before final thresholding
    resized_object_alpha_mask_1ch_dirty = cv2.resize(cropped_object_mask_1ch, (tw, th), interpolation=cv2.INTER_NEAREST)
    _, resized_object_alpha_mask_1ch = cv2.threshold(resized_object_alpha_mask_1ch_dirty, 127, 255, cv2.THRESH_BINARY)

    final_image = inpainted_bg.copy()
    # Define the exact slice for ROI to avoid off-by-one if tx+tw=W_bg or ty+th=H_bg
    roi = final_image[ty : min(ty + th, H_bg), tx : min(tx + tw, W_bg)]
    
    # Adjust resized object and mask if ROI is smaller due to boundary clipping (should not happen if tx,ty,tw,th are from valid boundingRect on aligned mask)
    # For safety, ensure alpha mask matches ROI dimensions if clipping occurred (though it shouldn't be necessary)
    # This step is complex if roi.shape is different from (th, tw). For now, we assume tx,ty,tw,th are valid.
    # If tx,ty,tw,th from boundingRect are correct, roi.shape will be (th, tw)
    # unless th or tw were such that ty+th > H_bg or tx+tw > W_bg, which boundingRect should prevent.

    if roi.shape[0] != th or roi.shape[1] != tw:
        # This case indicates an issue with tx,ty,tw,th or that the ROI slice was clipped.
        # We must ensure object and mask are resized to roi.shape[:2] inverted (W,H for cv2.resize) 
        th_roi, tw_roi = roi.shape[:2]
        if tw_roi == 0 or th_roi == 0: return final_image # ROI is empty, nothing to do
        resized_object_unmasked = cv2.resize(resized_object_unmasked, (tw_roi, th_roi))
        resized_object_alpha_mask_1ch_dirty = cv2.resize(cropped_object_mask_1ch, (tw_roi, th_roi), interpolation=cv2.INTER_NEAREST)
        _, resized_object_alpha_mask_1ch = cv2.threshold(resized_object_alpha_mask_1ch_dirty, 127, 255, cv2.THRESH_BINARY)
        # Update th, tw to match actual ROI dimensions
        th, tw = th_roi, tw_roi

    alpha_mask_inv = cv2.bitwise_not(resized_object_alpha_mask_1ch)
    
    # Ensure bg_part and fg_part are compatible with roi shape
    # roi is (h,w,c), masks are (h,w)
    bg_part = cv2.bitwise_and(roi, roi, mask=alpha_mask_inv[0:th, 0:tw])
    fg_part = cv2.bitwise_and(resized_object_unmasked, resized_object_unmasked, mask=resized_object_alpha_mask_1ch[0:th, 0:tw])
    
    combined_roi = cv2.add(bg_part, fg_part)

    final_image[ty:ty + th, tx:tx + tw] = combined_roi
    
    return final_image

# get args 
parser = argparse.ArgumentParser()
setup_args(parser)
args = parser.parse_args(sys.argv[1:])
# build models
model = {}
# build the sam model
model_type="vit_h"
ckpt_p=args.sam_ckpt
model_sam = sam_model_registry[model_type](checkpoint=ckpt_p)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_sam.to(device=device)
model['sam'] = SamPredictor(model_sam)

# build the lama model
lama_config = args.lama_config
lama_ckpt = args.lama_ckpt
device = "cuda" if torch.cuda.is_available() else "cpu"
model['lama'] = build_lama_model(lama_config, lama_ckpt, device=device)

button_size = (100,50)
with gr.Blocks() as demo:
    clicked_points = gr.State([])
    origin_image = gr.State(None)
    click_mask = gr.State(None)
    features = gr.State(None)
    orig_h = gr.State(None)
    orig_w = gr.State(None)
    input_h = gr.State(None)
    input_w = gr.State(None)

    # States for the new object image processing
    new_object_image_input = gr.State(None)
    new_object_features = gr.State(None)
    new_object_orig_h = gr.State(None)
    new_object_orig_w = gr.State(None)
    new_object_input_h = gr.State(None)
    new_object_input_w = gr.State(None)
    new_object_clicked_points = gr.State([])
    # This state will hold the actual mask data for the new object
    new_object_selected_mask = gr.State(None)

    with gr.Row():
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("## Input Image (Object to Remove/Replace)")
            with gr.Row():
                # img = gr.Image(label="Input Image")
                source_image_click = gr.Image(
                    type="numpy",
                    height=300, # Consider making height consistent or configurable
                    interactive=True,
                    label="Image: Upload an image and click the region you want to edit.",
                )
            
            with gr.Row():
                gr.Markdown("## New Object Image (Object to Add)")
            with gr.Row():
                new_object_uploader = gr.Image(
                    type="numpy",
                    height=300, # Match height for consistency
                    interactive=True,
                    label="Image: Upload image with the new object and click to select it.",
                )

            with gr.Row():
                point_prompt = gr.Radio(
                    choices=["Foreground Point", "Background Point"],
                    value="Foreground Point",
                    label="Point Label",
                    interactive=True,
                    show_label=False,
                )
                image_resolution = gr.Slider(
                    label="Image Resolution",
                    minimum=256,
                    maximum=768,
                    value=512,
                    step=64,
                )
                dilate_kernel_size = gr.Slider(label="Dilate Kernel Size", minimum=0, maximum=30, step=1, value=3)
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("## Control Panel")
            text_prompt = gr.Textbox(label="Text Prompt")
            lama = gr.Button("Inpaint Image", variant="primary")
            replace_sd = gr.Button("Replace Anything with SD", variant="primary")
            
            replace_obj_button = gr.Button("Replace with Object from Image", variant="primary") # New Button
            
            clear_button_image = gr.Button(value="Reset", label="Reset", variant="secondary")

    # todo: maybe we can delete this row, for it's unnecessary to show the original mask for customers
    # Row for mask displays and results
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Original Object Mask")
            with gr.Row():
                # This click_mask is a gr.Image component for display
                # The gr.State named click_mask holds its data
                click_mask_display = gr.Image(type="numpy", label="Mask for Original Object") 
        
        with gr.Column(): # New column for the new object's mask
            with gr.Row():
                gr.Markdown("## New Object Mask")
            with gr.Row():
                new_object_mask_display = gr.Image(type="numpy", label="Mask for New Object")

        with gr.Column():
            with gr.Row():
                gr.Markdown("## Image Removed with Mask (Inpainted)")
            with gr.Row():
                img_rm_with_mask = gr.Image(
                    type="numpy", label="Image Removed with Mask")
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Replace Anything with Mask")
            with gr.Row():
                img_replace_with_mask = gr.Image(
                    type="numpy", label="Image Replace Anything with Mask")

        with gr.Column(): # New column for the final composited image
            with gr.Row():
                gr.Markdown("## Image with New Object")
            with gr.Row():
                img_replaced_with_obj_display = gr.Image(type="numpy", label="Image with New Object Placed")

    source_image_click.upload(
        image_upload,
        inputs=[source_image_click, image_resolution],
        outputs=[origin_image, features, orig_h, orig_w, input_h, input_w],
    )
    source_image_click.select(
        process_image_click,
        inputs=[origin_image, point_prompt,
                clicked_points, image_resolution,
                features, orig_h, orig_w, input_h, input_w],
        # Output the mask data to both the State and the Image display component
        outputs=[source_image_click, clicked_points, click_mask_display, click_mask], 
        show_progress=True,
        queue=True,
    )

    # Setup for the new object image uploader
    new_object_uploader.upload(
        image_upload, # Reuse existing image_upload function
        inputs=[new_object_uploader, image_resolution],
        outputs=[new_object_image_input, new_object_features, 
                 new_object_orig_h, new_object_orig_w, 
                 new_object_input_h, new_object_input_w],
    )
    new_object_uploader.select(
        process_image_click, # Reuse existing process_image_click function
        inputs=[new_object_image_input, point_prompt, # Use same point_prompt
                new_object_clicked_points, image_resolution, # Use same image_resolution
                new_object_features, new_object_orig_h, new_object_orig_w, 
                new_object_input_h, new_object_input_w],
        # Output mask data to the new object's mask display and its state variable
        outputs=[new_object_uploader, new_object_clicked_points, 
                 new_object_mask_display, new_object_selected_mask],
        show_progress=True,
        queue=True,
    )

    lama.click(
        get_inpainted_img,
        [origin_image, click_mask, image_resolution],
        [img_rm_with_mask]
    )
    
    replace_sd.click(
        get_replace_img_with_sd,
        [origin_image, click_mask, image_resolution, text_prompt],
        [img_replace_with_mask]
    )

    # Handler for the new "Replace with Object from Image" button
    def run_replace_with_object_handler(orig_img, orig_mask, new_obj_img, new_obj_mask, img_res):
        if orig_img is None or orig_mask is None or new_obj_img is None or new_obj_mask is None:
            # Handle missing inputs, e.g., return a blank image or error message
            # For now, return a placeholder if we can't proceed.
            # User should be guided by UI to provide all inputs.
            return np.zeros((img_res, img_res, 3), dtype=np.uint8) 

        inpainted_image = get_inpainted_img(orig_img, orig_mask, img_res)
        final_image = composite_object_on_inpainted(inpainted_image, new_obj_img, new_obj_mask, orig_mask)
        return final_image

    replace_obj_button.click(
        run_replace_with_object_handler,
        inputs=[origin_image, click_mask, new_object_image_input, new_object_selected_mask, image_resolution],
        outputs=[img_replaced_with_obj_display]
    )

    def reset(*args):
        # Create a list of None values matching the number of arguments
        return [None for _ in args]

    # Update the reset button to clear new states and UI components
    clear_button_image.click(
        reset,
        # List all states and UI components that need to be reset
        [
            origin_image, features, click_mask, img_rm_with_mask, img_replace_with_mask, # Existing
            source_image_click, # Existing UI component to clear (already covered by origin_image=None effectively)
            
            # New states for new object image
            new_object_image_input, new_object_features,
            new_object_orig_h, new_object_orig_w,
            new_object_input_h, new_object_input_w,
            new_object_clicked_points, new_object_selected_mask,
            
            # New UI components to clear/reset
            new_object_uploader, # Image display for new object uploader
            click_mask_display, # Display for original mask
            new_object_mask_display, # Image display for new object mask
            img_replaced_with_obj_display # Image display for final composite
        ],
        [
            origin_image, features, click_mask, img_rm_with_mask, img_replace_with_mask,
            source_image_click, 
            
            new_object_image_input, new_object_features,
            new_object_orig_h, new_object_orig_w,
            new_object_input_h, new_object_input_w,
            new_object_clicked_points, new_object_selected_mask,
            
            new_object_uploader,
            click_mask_display,
            new_object_mask_display,
            img_replaced_with_obj_display
        ]
    )

if __name__ == "__main__":
    demo.queue(api_open=False).launch(server_name='0.0.0.0', share=False, debug=True)