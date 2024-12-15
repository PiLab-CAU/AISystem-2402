import numpy as np
from PIL import Image
import random
from typing import List
from .base import BaseAugmentation
from .noise import GaussianNoise, TextureDeformation, RandomErase, GlitchEffect
from .geometric import LocalDeformationAdvanced, RandomRotate
from .color import AdvancedColorDistortion

class AnomalyAugmenter:
    def __init__(self, severity: float = 0.7):
        self.augmentations = [
            AdvancedColorDistortion(severity),
            TextureDeformation(severity),
            LocalDeformationAdvanced(severity),
            RandomRotate(severity),
            GaussianNoise(severity),
            RandomErase(severity),
            GlitchEffect(severity)  # 새로운 augmentation 추가
        ]
        self.severity = severity
    
    def generate_anomaly(self, image: Image.Image) -> Image.Image:
        num_augs = random.randint(2, 4)
        selected_augs = random.sample(self.augmentations, num_augs)
        
        img = image
        for aug in selected_augs:
            aug.severity = self.severity * random.uniform(0.7, 1.5)
            img = aug(img)
            
        return img