from abc import ABC, abstractmethod


class UpscalerInterface(ABC):
    @abstractmethod
    def generate(self, image):
        raise NotImplementedError("generate method is not implemented")