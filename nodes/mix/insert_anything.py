import os
import cv2
import numpy as np
import torch
from PIL import Image
import folder_paths
from .utils import get_bbox_from_mask, expand_bbox, pad_to_square, box2squre, crop_back, expand_image_mask

class InsertAnythingNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "background_mask": ("MASK",),
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "flux_fill_pipe": ("MODEL",),
                "flux_redux_pipe": ("STYLE_MODEL",),
                "seed": ("INT", {"default": 666, "min": -1, "max": 999999999}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "expansion_ratio": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 3.0, "step": 0.1}),
                "mask_dilation": ("INT", {"default": 2, "min": 0, "max": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "insert_anything"
    CATEGORY = "image/inpainting"

    def __init__(self):
        pass
    
    def tensor_to_pil(self, image_tensor):
        """Convert a PyTorch tensor to a PIL Image."""
        # Convert to numpy and adjust dimensions if needed
        if len(image_tensor.shape) == 4 and image_tensor.shape[0] == 1:
            # Remove batch dimension if present with size 1
            image_tensor = image_tensor[0]
            
        # Convert to numpy, move channel dimension to the end
        if image_tensor.shape[0] in [1, 3, 4]:  # Channel-first format
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        else:  # Assume already in HWC format
            image_np = image_tensor.cpu().numpy()
            
        # Scale from [0,1] to [0,255] if needed
        if image_np.max() <= 1.0:
            image_np = (image_np * 255.0).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
            
        return Image.fromarray(image_np)
    
    def mask_to_pil(self, mask_tensor):
        """Convert a mask tensor to a PIL Image."""
        if len(mask_tensor.shape) == 3 and mask_tensor.shape[0] == 1:  # Single batch dimension
            mask_np = mask_tensor[0].cpu().numpy()
        else:
            mask_np = mask_tensor.cpu().numpy()
            
        # Scale from [0,1] to [0,255] if needed
        if mask_np.max() <= 1.0:
            mask_np = (mask_np * 255.0).astype(np.uint8)
        else:
            mask_np = mask_np.astype(np.uint8)
            
        return Image.fromarray(mask_np)
    
    def pil_to_tensor(self, pil_image):
        """Convert a PIL Image to a PyTorch tensor."""
        # Convert to numpy array
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        
        # Change from HWC to BCHW format
        if len(img_np.shape) == 3:
            img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
        
        # Add batch dimension
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        
        return img_tensor

    def insert_anything(self, background_image, background_mask, reference_image, reference_mask, 
                       flux_fill_pipe, flux_redux_pipe, seed, steps, guidance_scale,
                       expansion_ratio=1.3, mask_dilation=2):
        
        # Use the provided models directly
        self.pipe = flux_fill_pipe
        self.redux = flux_redux_pipe
        
        # Convert inputs to PIL
        tar_image = self.tensor_to_pil(background_image)
        tar_mask = self.mask_to_pil(background_mask)
        ref_image = self.tensor_to_pil(reference_image)
        ref_mask = self.mask_to_pil(reference_mask)

        # Convert to RGB/L
        tar_image = tar_image.convert("RGB")
        tar_mask = tar_mask.convert("L")
        ref_image = ref_image.convert("RGB")
        ref_mask = ref_mask.convert("L")

        # Convert to numpy arrays
        tar_image = np.asarray(tar_image)
        tar_mask = np.asarray(tar_mask)
        tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)

        ref_image = np.asarray(ref_image)
        ref_mask = np.asarray(ref_mask)
        ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

        # Check if masks are valid
        if tar_mask.sum() == 0:
            raise Exception('No mask for the background image. Please check the mask!')

        if ref_mask.sum() == 0:
            raise Exception('No mask for the reference image. Please check the mask!')

        # Process reference image
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)
        y1, y2, x1, x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
        ref_mask = ref_mask[y1:y2, x1:x2]
        
        # Expand reference image and mask
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=expansion_ratio)
        masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)

        # Process target mask
        if mask_dilation > 0:
            kernel = np.ones((7, 7), np.uint8)
            tar_mask = cv2.dilate(tar_mask, kernel, iterations=mask_dilation)

        # Crop target image
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)
        tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=2)
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
        y1, y2, x1, x2 = tar_box_yyxx_crop

        old_tar_image = tar_image.copy()
        tar_image = tar_image[y1:y2, x1:x2, :]
        tar_mask = tar_mask[y1:y2, x1:x2]

        H1, W1 = tar_image.shape[0], tar_image.shape[1]

        # Prepare images for processing
        size = (768, 768)
        tar_mask = pad_to_square(tar_mask, pad_value=0)
        tar_mask = cv2.resize(tar_mask, size)

        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)
        
        # Generate prior
        pipe_prior_output = self.redux(Image.fromarray(masked_ref_image))

        tar_image = pad_to_square(tar_image, pad_value=255)
        H2, W2 = tar_image.shape[0], tar_image.shape[1]
        tar_image = cv2.resize(tar_image, size)

        # Create diptych
        diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)
        tar_mask = np.stack([tar_mask, tar_mask, tar_mask], -1)
        mask_black = np.ones_like(tar_image) * 0
        mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)

        diptych_ref_tar = Image.fromarray(diptych_ref_tar)
        mask_diptych[mask_diptych == 1] = 255
        mask_diptych = Image.fromarray(mask_diptych)

        # Generate image
        generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        edited_image = self.pipe(
            image=diptych_ref_tar,
            mask_image=mask_diptych,
            height=mask_diptych.size[1],
            width=mask_diptych.size[0],
            max_sequence_length=512,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **pipe_prior_output,
        ).images[0]

        # Crop result
        width, height = edited_image.size
        left = width // 2
        right = width
        top = 0
        bottom = height
        edited_image = edited_image.crop((left, top, right, bottom))

        # Crop back to original size
        edited_image = np.array(edited_image)
        edited_image = crop_back(edited_image, old_tar_image, 
                               np.array([H1, W1, H2, W2]), np.array(tar_box_yyxx_crop))
        edited_image = Image.fromarray(edited_image)

        # Convert back to tensor
        result_tensor = self.pil_to_tensor(edited_image)

        return (result_tensor,)
