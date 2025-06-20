import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as transforms
from torchvision.utils import make_grid
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from transformers import SiglipImageProcessor, SiglipVisionModel
from huggingface_hub import hf_hub_download

import comfy.model_management as mm
from comfy.utils import ProgressBar

from .model import create_model


# Custom transform to pad images to square
class PadToSquare:
    def __call__(self, img):
        _, h, w = img.shape
        max_side = max(h, w)
        pad_h = (max_side - h) // 2
        pad_w = (max_side - w) // 2
        padding = (pad_w, pad_h, max_side - w - pad_w, max_side - h - pad_h)
        return transforms.functional.pad(img, padding, padding_mode="edge")


class TryOffDiffLoaderNode:
    """
    Node for loading TryOffDiff models and components
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["upper", "lower", "dress", "multi"], {"default": "multi"}),
                "repo_id": ("STRING", {"default": "rizavelioglu/tryoffdiff"}),
                "force_download": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("TRYOFFDIFF_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "TryOffDiff"
    TITLE = "TryOffDiff Model Loader"
    
    def load_pipeline(self, model_type, repo_id, force_download):
        device = mm.get_torch_device()
        
        # Model configurations
        model_paths = {
            "upper": {"class_name": "TryOffDiffv2Single", "path": "tryoffdiffv2_upper.pth"},
            "lower": {"class_name": "TryOffDiffv2Single", "path": "tryoffdiffv2_lower.pth"},
            "dress": {"class_name": "TryOffDiffv2Single", "path": "tryoffdiffv2_dress.pth"},
            "multi": {"class_name": "TryOffDiffv2", "path": "tryoffdiffv2_multi.pth"},
        }

        # check if folder tryoffdiff exists, if not create it
        if not os.path.exists("./models/tryoffdiff"):
            os.makedirs("./models/tryoffdiff")
        
        # Load main model
        model_config = model_paths[model_type]
        path_model = hf_hub_download(
            repo_id=repo_id, 
            filename=model_config["path"], 
            force_download=force_download,
            local_dir="./models/tryoffdiff"
        )
        state_dict = torch.load(path_model, weights_only=True, map_location=device)
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model = create_model(model_config["class_name"]).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        # Load image encoder
        img_enc_transform = transforms.Compose([
            PadToSquare(),
            transforms.Resize((512, 512)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        
        ckpt = "google/siglip-base-patch16-512"
        img_processor = SiglipImageProcessor.from_pretrained(
            ckpt, do_resize=False, do_rescale=False, do_normalize=False
        )
        img_enc = SiglipVisionModel.from_pretrained(ckpt).eval().to(device)
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval().to(device)
        
        # Load scheduler
        scheduler_path = hf_hub_download(
            repo_id=repo_id, 
            filename="scheduler/scheduler_config_v2.json", 
            force_download=force_download
        )
        scheduler = EulerDiscreteScheduler.from_pretrained(scheduler_path)
        scheduler.is_scale_input_called = True
        
        pipeline = {
            "model": model,
            "model_type": model_type,
            "img_enc": img_enc,
            "img_processor": img_processor,
            "img_enc_transform": img_enc_transform,
            "vae": vae,
            "scheduler": scheduler,
            "device": device,
        }
        
        return (pipeline,)


class TryOffDiffNode:
    """
    Main TryOffDiff generation node
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("TRYOFFDIFF_PIPELINE",),
                "reference_image": ("IMAGE",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 20, "min": 5, "max": 1000, "step": 1}),
            },
            "optional": {
                "garment_types": (["Upper-Body", "Lower-Body", "Dress"], {"default": ["Upper-Body"]}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate"
    CATEGORY = "TryOffDiff"
    TITLE = "TryOffDiff Generator"
    
    def generate(self, pipeline, reference_image, seed, guidance_scale, num_inference_steps, garment_types=None):
        model = pipeline["model"]
        model_type = pipeline["model_type"]
        img_enc = pipeline["img_enc"]
        img_processor = pipeline["img_processor"]
        img_enc_transform = pipeline["img_enc_transform"]
        vae = pipeline["vae"]
        scheduler = pipeline["scheduler"]
        device = pipeline["device"]
        
        # Convert ComfyUI image format to PIL
        # ComfyUI images are in format [batch, height, width, channels] with values 0-1
        ref_img_np = (reference_image[0].cpu().numpy() * 255).astype(np.uint8)
        ref_img_pil = Image.fromarray(ref_img_np)
        
        # Save temp image for processing
        temp_path = "/tmp/ref_image.jpg"
        ref_img_pil.save(temp_path)
        
        # Process based on model type
        if model_type == "multi":
            if garment_types is None:
                garment_types = ["Upper-Body"]
            result = self._generate_multi(
                model, temp_path, garment_types, seed, guidance_scale, 
                num_inference_steps, img_enc, img_processor, img_enc_transform, 
                vae, scheduler, device
            )
        else:
            result = self._generate_single(
                model, temp_path, seed, guidance_scale, num_inference_steps,
                img_enc, img_processor, img_enc_transform, vae, scheduler, device
            )
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Convert result back to ComfyUI format
        result_np = np.array(result).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)  # Add batch dimension
        
        return (result_tensor,)
    
    def _generate_multi(self, model, input_image_path, garment_types, seed, guidance_scale, 
                       num_inference_steps, img_enc, img_processor, img_enc_transform, 
                       vae, scheduler, device):
        
        label_map = {"Upper-Body": 0, "Lower-Body": 1, "Dress": 2}
        valid_single = ["Upper-Body", "Lower-Body", "Dress"]
        valid_tuple = ["Upper-Body", "Lower-Body"]
        
        if not garment_types:
            garment_types = ["Upper-Body"]
        
        if len(garment_types) == 1 and garment_types[0] in valid_single:
            selected, label_indices = garment_types, [label_map[garment_types[0]]]
        elif sorted(garment_types) == sorted(valid_tuple):
            selected, label_indices = valid_tuple, [label_map[t] for t in valid_tuple]
        else:
            selected, label_indices = ["Upper-Body"], [label_map["Upper-Body"]]
        
        batch_size = len(selected)
        scheduler.set_timesteps(num_inference_steps)
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(batch_size, 4, 64, 64, generator=generator, device=device)
        
        # Process inputs
        from torchvision.io import read_image
        cond_image = img_enc_transform(read_image(input_image_path))
        inputs = {k: v.to(device) for k, v in img_processor(images=cond_image, return_tensors="pt").items()}
        cond_emb = img_enc(**inputs).last_hidden_state.to(device)
        cond_emb = cond_emb.expand(batch_size, *cond_emb.shape[1:])
        uncond_emb = torch.zeros_like(cond_emb) if guidance_scale > 1 else None
        label = torch.tensor(label_indices, device=device, dtype=torch.int64)
        
        # Progress bar
        pbar = ProgressBar(num_inference_steps)

        with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu'):
            for i, t in enumerate(scheduler.timesteps):
                t = t.to(device)
                if guidance_scale > 1:
                    noise_pred = model(torch.cat([x] * 2), t, torch.cat([uncond_emb, cond_emb]), torch.cat([label, label])).chunk(2)
                    noise_pred = noise_pred[0] + guidance_scale * (noise_pred[1] - noise_pred[0])
                else:
                    noise_pred = model(x, t, cond_emb, label)
                
                scheduler_output = scheduler.step(noise_pred, t, x)
                x = scheduler_output.prev_sample
                pbar.update(1)
        
        # Decode predictions from latent space
        decoded = vae.decode(1 / vae.config.scaling_factor * scheduler_output.pred_original_sample).sample
        images = (decoded / 2 + 0.5).cpu()
        grid = make_grid(images, nrow=len(images), normalize=True, scale_each=True)
        output_image = transforms.ToPILImage()(grid)
        
        return output_image
    
    def _generate_single(self, model, input_image_path, seed, guidance_scale, 
                        num_inference_steps, img_enc, img_processor, img_enc_transform, 
                        vae, scheduler, device):
        
        scheduler.set_timesteps(num_inference_steps)
        scheduler.timesteps = scheduler.timesteps.to(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(1, 4, 64, 64, generator=generator, device=device)
        
        # Process input image
        from torchvision.io import read_image
        cond_image = img_enc_transform(read_image(input_image_path))
        inputs = {k: v.to(device) for k, v in img_processor(images=cond_image, return_tensors="pt").items()}
        cond_emb = img_enc(**inputs).last_hidden_state.to(device)
        uncond_emb = torch.zeros_like(cond_emb) if guidance_scale > 1 else None
        
        # Progress bar
        pbar = ProgressBar(num_inference_steps)
        
        with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu'):
            for i, t in enumerate(scheduler.timesteps):
                t = t.to(device)
                if guidance_scale > 1:
                    noise_pred = model(torch.cat([x] * 2), t, torch.cat([uncond_emb, cond_emb])).chunk(2)
                    noise_pred = noise_pred[0] + guidance_scale * (noise_pred[1] - noise_pred[0])
                else:
                    noise_pred = model(x, t, cond_emb)
                
                scheduler_output = scheduler.step(noise_pred, t, x)
                x = scheduler_output.prev_sample
                pbar.update(1)
        
        # Decode predictions from latent space
        decoded = vae.decode(1 / vae.config.scaling_factor * scheduler_output.pred_original_sample).sample
        images = (decoded / 2 + 0.5).cpu()
        grid = make_grid(images, nrow=len(images), normalize=True, scale_each=True)
        output_image = transforms.ToPILImage()(grid)
        
        return output_image