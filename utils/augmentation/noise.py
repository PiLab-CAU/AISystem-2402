import numpy as np
from PIL import Image
from .base import BaseAugmentation

class GaussianNoise(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image).astype(np.float32)
        noise = np.random.normal(0, self.severity * 50, img_np.shape)
        noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
