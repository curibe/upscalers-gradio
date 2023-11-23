import importlib

import torch

from config.registers import register_upscaler
from config.schemas import RealESRGANParams, UpscalerMethodEnum
from upscalers.interface import UpscalerInterface


@register_upscaler
class RealESRGANUpscaler(UpscalerInterface):
    name = UpscalerMethodEnum.REALESRGAN.value

    def __init__(self, params: RealESRGANParams):
        # set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # import Pipeline
        self.module = importlib.import_module("RealESRGAN")
        self.pipeline = getattr(self.module, "RealESRGAN")
        # Get params
        self.scale = params.scale
        # Init/load model
        self.model = self.pipeline(self.device, scale=self.scale)
        self.model.load_weights('weights/RealESRGAN_x4.pth', download=True)

    def generate(self, image):
        upscaled_image = self.model.predict(image)
        return upscaled_image
