import numpy as np
from PIL import Image
from typing import List
from .base import BaseAugmentation
from .noise import GaussianNoise
from .geometric import LocalDeformation
from .color import ColorDistortion
from .randomrotate import RandomRotation
from .blur import RandomBlur
from .reddot import RedDotAnomaly
from .scartch import RandomScratch

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
            GaussianNoise(severity * 1.4),  
            LocalDeformation(severity * 1.2),  
            RandomDeletion(severity * 1.1),     
            RandomRotation(severity * 1.6), 
            RandomBlur(severity * 1.7),    
            RedDotAnomaly(severity * 1.5),
            RandomScratch(severity * 1.3)
        ]
    
    def generate_anomaly(self, image: Image.Image) -> Image.Image:
        # 최소 2개, 최대 4개의 augmentation 적용
        num_augs = np.random.randint(2, 5)
        # 더 다양한 조합의 이상치 생성을 위해 weight 부여
        weights = [1.5, 1.2, 1.0, 1.0, 1.0, 1.3, 1.4]  # 각 augmentation에 대한 가중치
        weights = np.array(weights) / sum(weights)
        
        selected_augs = np.random.choice(
            self.augmentations, 
            num_augs, 
            replace=False,
            p=weights
        )
        
        img = image
        for aug in selected_augs:
            img = aug(img)
            
        return img