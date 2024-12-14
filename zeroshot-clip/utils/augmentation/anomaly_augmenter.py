import numpy as np
from PIL import Image
from typing import List
from .base import BaseAugmentation
from .noise import GaussianNoise
from .geometric import LocalDeformation
from .color import ColorDistortion
from .randomrotate import RandomRotation
from .blur import RandomBlur

class RandomDeletion(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        height, width = img_np.shape[:2]
        
        num_patches = np.random.randint(1, 4)
        for _ in range(num_patches):
            x = np.random.randint(0, width - width//4)
            y = np.random.randint(0, height - height//4)
            patch_w = np.random.randint(width//8, width//4)
            patch_h = np.random.randint(height//8, height//4)
            img_np[y:y+patch_h, x:x+patch_w] = 0
            
        return Image.fromarray(img_np)

class AnomalyAugmenter:
    def __init__(self, severity: float = 0.7):
        self.augmentations: List[BaseAugmentation] = [
            GaussianNoise(severity * 1.2),
            LocalDeformation(severity * 1.3),
            ColorDistortion(severity * 1.1),
            RandomDeletion(severity * 1.2),
            RandomRotation(severity * 1.5),
            RandomBlur(severity * 1.5),
        ]
    
    def generate_anomaly(self, image: Image.Image) -> Image.Image:
        # Generate anomaly images by combining multiple augmentations
        num_augs = np.random.randint(2, 4)
        selected_augs = np.random.choice(self.augmentations, num_augs, replace=False)
        
        img = image
        for aug in selected_augs:
            img = aug(img)
            
        return img