import numpy as np
from PIL import Image
from .base import BaseAugmentation

class RandomBlur(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        from PIL import ImageFilter
        radius = self.severity * 2
        return image.filter(ImageFilter.GaussianBlur(radius=radius))