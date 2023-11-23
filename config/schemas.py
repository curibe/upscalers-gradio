from enum import Enum
from typing import Union

from pydantic import BaseModel
from .registers import register_params_model


class UpscalerMethodEnum(str, Enum):
    REALESRGAN = "RealESRGAN"
    BSRGAN = "BSRGAN"
    LATENTX2 = "Latent x2"
    LDMSUPERRESOL4X = "LDM Super Resolution 4x"
    SD4X = "SD 4x"


@register_params_model(UpscalerMethodEnum.REALESRGAN.value)
class RealESRGANParams(BaseModel):
    scale: int = 2


@register_params_model(UpscalerMethodEnum.LDMSUPERRESOL4X.value)
class LDMSuperResolutionParams(BaseModel):
    num_inference_steps: int = 10
    eta: int = 1


UpscalerParams = Union[RealESRGANParams,LDMSuperResolutionParams]