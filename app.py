import sys
import os
import gradio as gr
import numpy as np
from PIL import Image
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from src.shadow_inference import ShadowRemover, get_available_models

PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "ShadowDiffusion", "pretrained")
MODEL_CACHE = {}


def load_model(model_name: str) -> Optional[ShadowRemover]:
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    
    models = get_available_models(PRETRAINED_DIR)
    if model_name not in models:
        return None
    
    remover = ShadowRemover(model_path=models[model_name])
    MODEL_CACHE[model_name] = remover
    return remover


def remove_shadow(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    model_name: str,
    num_steps: int,
    show_mask: bool
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    if image is None:
        return None, None, "Please upload an image."
    
    remover = load_model(model_name)
    if remover is None:
        return None, None, f"Model '{model_name}' not found. Please download pretrained weights."
    
    pil_image = Image.fromarray(image)
    pil_mask = Image.fromarray(mask) if mask is not None else None
    
    original_size = pil_image.size
    
    shadow_removed, estimated_mask = remover.remove_shadow(
        pil_image, pil_mask, num_steps=num_steps
    )
    
    result_pil = Image.fromarray(shadow_removed)
    result_pil = result_pil.resize(original_size, Image.BILINEAR)
    result = np.array(result_pil)
    
    mask_out = None
    if show_mask:
        mask_pil = Image.fromarray(estimated_mask)
        mask_pil = mask_pil.resize(original_size, Image.BILINEAR)
        mask_out = np.array(mask_pil)
    
    return result, mask_out, "Inference complete."


def get_model_choices():
    models = get_available_models(PRETRAINED_DIR)
    if models:
        return list(models.keys())
    return ["SRD", "ISTD", "ISTD_plus"]


def create_ui():
    with gr.Blocks(title="ShadowDiffusion - Shadow Removal") as app:
        gr.Markdown("# ShadowDiffusion Shadow Removal")
        gr.Markdown("Upload a shadow image to remove shadows using the ShadowDiffusion model (CVPR 2023).")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image (with shadow)", type="numpy")
                input_mask = gr.Image(label="Shadow Mask (optional)", type="numpy")
                
                model_dropdown = gr.Dropdown(
                    choices=get_model_choices(),
                    value="SRD",
                    label="Model"
                )
                
                steps_slider = gr.Slider(
                    minimum=5,
                    maximum=25,
                    value=5,
                    step=5,
                    label="DDIM Sampling Steps"
                )
                
                show_mask_checkbox = gr.Checkbox(
                    label="Show Estimated Shadow Mask",
                    value=True
                )
                
                run_button = gr.Button("Remove Shadow", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Shadow-Removed Image", type="numpy")
                output_mask = gr.Image(label="Estimated Shadow Mask", type="numpy")
                status_text = gr.Textbox(label="Status", interactive=False)
        
        run_button.click(
            fn=remove_shadow,
            inputs=[input_image, input_mask, model_dropdown, steps_slider, show_mask_checkbox],
            outputs=[output_image, output_mask, status_text]
        )
    
    return app


if __name__ == "__main__":
    models = get_available_models(PRETRAINED_DIR)
    if not models:
        print("WARNING: No pretrained models found.")
        print(f"Please download weights from:")
        print("https://drive.google.com/file/d/12VV5HzxlIg_kCjMHsw8R9n543wYfGfAB/view?usp=sharing")
        print(f"And extract to: {PRETRAINED_DIR}")
        print()
    
    app = create_ui()
    app.launch(server_name="127.0.0.1", server_port=7860)
