from enum import Enum
from typing import Union

from pydantic import BaseModel

from .registers import register_params_model


class UpscalerMethodEnum(str, Enum):
    REALESRGAN = "RealESRGAN"
    BSRGAN = "BSRGAN"
    LDMSUPERRESOL4X = "LDM Super Resolution 4x"
    SD4X = "SD 4x"


@register_params_model(UpscalerMethodEnum.REALESRGAN.value)
class RealESRGANParams(BaseModel):
    scale: int = 2


@register_params_model(UpscalerMethodEnum.LDMSUPERRESOL4X.value)
class LDMSuperResolutionParams(BaseModel):
    num_inference_steps: int = 10
    eta: int = 1


@register_params_model(UpscalerMethodEnum.BSRGAN.value)
class BSRGANParams(BaseModel):
    scale: int = 4


@register_params_model(UpscalerMethodEnum.SD4X.value)
class SD4XParams(BaseModel):
    prompt: str
    num_steps: int = 20
    guidance_scale: float = 0


UpscalerParams = Union[RealESRGANParams, LDMSuperResolutionParams, BSRGANParams, SD4XParams]
