from typing import Dict, Type

from upscalers import UpscalerInterface

upscalers: Dict[str, Type[UpscalerInterface]] = {}
params_models = {}

def register_upscaler(cls):
    upscalers[cls.name] = cls
    return cls


def register_params_model(name):
    def decorator(cls):
        params_models[name] = cls
        return cls
    return decorator
