import numpy as np
import random
from PIL import Image
from typing import List
from .base import BaseAugmentation
from .scratch import SpotScratchCrack
from .rotation import Rotate

class RandomDeletion(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        height, width = img_np.shape[:2]

        num_patches = np.random.randint(1, 4)
        for _ in range(num_patches):
            x = np.random.randint(0, width - width // 4)
            y = np.random.randint(0, height - height // 4)
            patch_w = np.random.randint(width // 8, width // 4)
            patch_h = np.random.randint(height // 8, height // 4)

            # Randomly select a color: black, white, red, green, or blue
            color_options = [
                (0, 0, 0),     # Black
                (255, 255, 255) # White
            ]
            color = color_options[np.random.randint(len(color_options))]

            # Apply the selected color to the patch
            img_np[y:y+patch_h, x:x+patch_w] = color

        return Image.fromarray(img_np)
    
class RandomColorDeletion(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        height, width = img_np.shape[:2]

        num_patches = np.random.randint(1, 4)
        for _ in range(num_patches):
            x = np.random.randint(0, width - width // 4)
            y = np.random.randint(0, height - height // 4)
            patch_w = np.random.randint(width // 8, width // 4)
            patch_h = np.random.randint(height // 8, height // 4)

            # Randomly select a color: black, white, red, green, or blue
            color_options = [
                (255, 0, 0),   # Red
                (0, 255, 0),   # Green
                (0, 0, 255)    # Blue
            ]
            color = color_options[np.random.randint(len(color_options))]

            # Apply the selected color to the patch
            img_np[y:y+patch_h, x:x+patch_w] = color

        return Image.fromarray(img_np)

class AnomalyAugmenter:
    def __init__(self, severity: float = 0.7):
        self.augmentations: List[BaseAugmentation] = [
            SpotScratchCrack(severity),
            RandomColorDeletion(severity),
            Rotate(severity),
            RandomDeletion(severity)
        ]
    
    def generate_anomaly(self, image: Image.Image) -> Image.Image:
        # Generate anomaly images by combining multiple augmentations
        np.random.seed(42)
        random.seed(42)
        
        num_augs = np.random.randint(2, 4)
        selected_augs = np.random.choice(self.augmentations, num_augs, replace=False)
        augmentations = []
        img = image
        for aug in selected_augs:
            img = aug(img)
            augmentations.append(type(aug).__name__)
        return img, augmentations
