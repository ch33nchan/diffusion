import sys
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import json
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ShadowDiffusion'))

from model.sr3_modules import diffusion as diffusion_module
from model.sr3_modules import unet


class ShadowRemover:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        image_size: int = 256
    ):
        self.device = device
        self.image_size = image_size
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: str) -> nn.Module:
        unet_model = unet.UNet(
            in_channel=7,
            out_channel=3,
            norm_groups=16,
            inner_channel=64,
            channel_mults=[1, 2, 4, 8],
            attn_res=[16],
            res_blocks=2,
            dropout=0,
            image_size=self.image_size
        )
        
        model = diffusion_module.GaussianDiffusion(
            denoise_fn=unet_model,
            image_size=self.image_size,
            channels=3,
            loss_type='l1',
            conditional=True,
            schedule_opt=None
        )
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        
        return model

    def set_sampling_steps(self, t_sampling: int = 5):
        schedule_opt = {
            'schedule': 'linear',
            'n_timestep': 1000,
            'linear_start': 1e-4,
            'linear_end': 0.02,
            'T_sampling': t_sampling
        }
        self.model.set_new_noise_schedule(schedule_opt, self.device)

    def preprocess(
        self, 
        image: Image.Image, 
        mask: Optional[Image.Image] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        img_np = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        img_tensor = img_tensor * 2.0 - 1.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        if mask is not None:
            mask = mask.convert("L")
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_np = np.array(mask).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        else:
            mask_tensor = torch.ones(1, 1, self.image_size, self.image_size)
        
        mask_tensor = mask_tensor.to(self.device)
        
        return img_tensor, mask_tensor

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.squeeze().cpu()
        tensor = (tensor + 1.0) / 2.0
        tensor = tensor.clamp(0, 1)
        img_np = tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        return img_np

    @torch.no_grad()
    def remove_shadow(
        self, 
        image: Image.Image, 
        mask: Optional[Image.Image] = None,
        num_steps: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.set_sampling_steps(num_steps)
        
        img_tensor, mask_tensor = self.preprocess(image, mask)
        
        sr_result, estimated_mask = self.model.super_resolution(
            img_tensor, mask_tensor, continous=False
        )
        
        shadow_removed = self.postprocess(sr_result)
        mask_out = self.postprocess(estimated_mask)
        
        return shadow_removed, mask_out


def get_available_models(pretrained_dir: str) -> dict:
    models = {}
    if os.path.exists(pretrained_dir):
        for dataset in ['SRD', 'ISTD', 'ISTD_plus']:
            dataset_dir = os.path.join(pretrained_dir, dataset)
            if os.path.exists(dataset_dir):
                for f in os.listdir(dataset_dir):
                    if f.endswith('_gen.pth'):
                        models[dataset] = os.path.join(dataset_dir, f)
    return models
