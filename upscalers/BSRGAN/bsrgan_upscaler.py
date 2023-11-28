from pathlib import Path

import numpy as np
import requests
import torch

from config.registers import register_upscaler
from config.schemas import BSRGANParams, UpscalerMethodEnum
from upscalers.interface import UpscalerInterface
from .BSRGAN.models.network_rrdbnet import RRDBNet as net
from .BSRGAN.utils import utils_image as util

PARENT_DIR = Path(__file__).resolve().parent / "BSRGAN"
MODEL_DIR = "model_zoo"
MODEL_NAME = {4: "BSRGAN.pth", 2: "BSRGANx2.pth"}


@register_upscaler
class BSRGAN(UpscalerInterface):
    name = UpscalerMethodEnum.BSRGAN.value

    def __init__(self, params=BSRGANParams):
        self.scale_factor = params.scale
        self.model_path = PARENT_DIR / MODEL_DIR / MODEL_NAME[self.scale_factor]
        self.model_url = f"https://github.com/cszn/KAIR/releases/download/v1.0/{MODEL_NAME[self.scale_factor]}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = self._init_model()

    def _download_model(self):
        if self.model_path.exists():
            print(f"Model file '{self.model_path.name}' already exists. Skipping download.")
        else:
            print(f"Downloading model: {self.model_path.name} ...")
            r = requests.get(self.model_url, allow_redirects=True)
            open(self.model_path, "wb").write(r.content)
            print("Download complete!")

    def _init_model(self):
        self._download_model()
        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=self.scale_factor)
        model.load_state_dict(torch.load(self.model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        return model.to(self.device)

    def generate(self, image):
        #  Load the image
        base_image = np.array(image)

        # Convert to tensor
        tensor_image = util.uint2tensor4(base_image)
        tensor_image = tensor_image.to(self.device)

        # Upscale the image
        upscaled_image = self.pipeline(tensor_image)
        upscaled_image = util.tensor2uint(upscaled_image)

        # Clear the cache
        torch.cuda.empty_cache()

        return upscaled_image
