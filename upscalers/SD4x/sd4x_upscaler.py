import torch
from diffusers import StableDiffusionUpscalePipeline

from config.registers import register_upscaler
from config.schemas import SD4XParams
from upscalers.interface import UpscalerInterface


@register_upscaler
class SD4xUpscaler(UpscalerInterface):
    def __init__(self, params: SD4XParams):
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_id = "stabilityai/stable-diffusion-x4-upscaler"
        self.pipe = StableDiffusionUpscalePipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(self.device)

    def generate(self, image):
        upscaled_image = self.pipe(
            prompt=self.params.prompt,
            image=image,
            num_inference_steps=self.params.num_inference_steps,
            guidance_scale=self.params.guidance_scale,
        ).images[0]
        return upscaled_image
