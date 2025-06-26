from enum import Enum, unique
from typing import Any

import torch
import torchvision.transforms.v2 as transforms
from diffusers import UNet2DConditionModel
from torch import nn


class TryOffDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.proj = nn.Linear(1024, 77)
        self.norm = nn.LayerNorm(768)

    def forward(self, noisy_latents, t, cond_emb):
        cond_emb = self.transformer(cond_emb)
        cond_emb = self.proj(cond_emb.transpose(1, 2))
        cond_emb = self.norm(cond_emb.transpose(1, 2))
        return self.unet(noisy_latents, t, encoder_hidden_states=cond_emb).sample

class TryOffDiffv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(320, 640, 1280, 1280),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=768,
            class_embed_type=None,
            num_class_embeds=3,
        )
        # Load the pretrained weights into the custom model, skipping incompatible keys
        pretrained_state_dict = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").state_dict()
        self.unet.load_state_dict(pretrained_state_dict, strict=False)

        self.proj = nn.Linear(1024, 77)
        self.norm = nn.LayerNorm(768)

    def forward(self, noisy_latents, t, cond_emb, class_labels):
        cond_emb = self.proj(cond_emb.transpose(1, 2))
        cond_emb = self.norm(cond_emb.transpose(1, 2))
        return self.unet(noisy_latents, t, encoder_hidden_states=cond_emb, class_labels=class_labels).sample

class TryOffDiffv2Single(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        self.proj = nn.Linear(1024, 77)
        self.norm = nn.LayerNorm(768)

    def forward(self, noisy_latents, t, cond_emb):
        cond_emb = self.proj(cond_emb.transpose(1, 2))
        cond_emb = self.norm(cond_emb.transpose(1, 2))
        return self.unet(noisy_latents, t, encoder_hidden_states=cond_emb).sample

@unique
class ModelName(Enum):
    TryOffDiff = TryOffDiff
    TryOffDiffv2 = TryOffDiffv2
    TryOffDiffv2Single = TryOffDiffv2Single

def create_model(model_name: str, **kwargs: Any) -> Any:
    model_class = ModelName[model_name].value
    return model_class(**kwargs)