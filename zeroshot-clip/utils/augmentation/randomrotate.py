import numpy as np
from PIL import Image
from .base import BaseAugmentation

class RandomRotation(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        angle = np.random.uniform(-30 * self.severity, 30 * self.severity)
        return image.rotate(angle, expand=True)