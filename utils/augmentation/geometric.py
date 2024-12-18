import numpy as np
from PIL import Image
from .base import BaseAugmentation

class LocalDeformation(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        x1 = np.random.randint(0, width // 2)
        y1 = np.random.randint(0, height // 2)
        x2 = x1 + width // 3
        y2 = y1 + height // 3
        
        img_np = np.array(image)
        region = img_np[y1:y2, x1:x2]
        distorted = np.roll(region, shift=int(self.severity * 20))
        img_np[y1:y2, x1:x2] = distorted
        
        return Image.fromarray(img_np)
