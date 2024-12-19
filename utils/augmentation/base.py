from abc import ABC, abstractmethod
from PIL import Image

class BaseAugmentation(ABC):
    def __init__(self, severity: float = 0.7):
        self.severity = severity
    
    @abstractmethod
    def __call__(self, image: Image.Image) -> Image.Image:
        pass
