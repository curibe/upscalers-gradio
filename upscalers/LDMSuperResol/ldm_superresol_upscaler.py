import importlib

import torch

from config.schemas import LDMSuperResolutionParams, UpscalerMethodEnum
from config.registers import register_upscaler
from upscalers.interface import UpscalerInterface


@register_upscaler
class LDMSuperResol4XUpscaler(UpscalerInterface):
    name = UpscalerMethodEnum.LDMSUPERRESOL4X.value

    def __init__(self, params: LDMSuperResolutionParams):
        # set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # import Pipeline
        self.module = importlib.import_module("diffusers")
        self.Pipeline = getattr(self.module,"LDMSuperResolutionPipeline")
        # Get params
        self.params = params
        # Set Model id
        self.model_id = "CompVis/ldm-super-resolution-4x-openimages"
        # load model and scheduler
        self.pipe = self.Pipeline.from_pretrained(self.model_id)

        self.pipe = self.pipe.to(self.device)

    def generate(self, image):
        upscaled_image = self.pipe(image, num_inference_steps=self.params.num_inference_steps, eta=self.params.eta).images[0]
        return upscaled_image
