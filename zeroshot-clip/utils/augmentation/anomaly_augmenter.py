import numpy as np
from PIL import Image
import random
from typing import List
from .base import BaseAugmentation
from .noise import GaussianNoise, TextureDeformation
from .geometric import LocalDeformationAdvanced, RandomRotate
from .color import AdvancedColorDistortion

class AnomalyAugmenter:
    def __init__(self, severity: float = 0.7):
        """
        Initialize advanced anomaly augmenter.
        
        Args:
            severity: Base severity level for augmentations (default: 0.7)
        """
        self.augmentations = [
            AdvancedColorDistortion(severity),
            TextureDeformation(severity),
            LocalDeformationAdvanced(severity),
            RandomRotate(severity),
            GaussianNoise(severity)  # GaussianNoise 추가
        ]
        self.severity = severity
    
    def generate_anomaly(self, image: Image.Image) -> Image.Image:
        """
        Generate anomaly with improved strategy.
        
        Args:
            image: Input image
            
        Returns:
            Image.Image: Augmented anomaly image
        """
        # Randomly select number of augmentations (2-3)
        num_augs = random.randint(2, 3)
        
        # Randomly select augmentations with adaptive severity
        selected_augs = random.sample(self.augmentations, num_augs)
        
        # Apply augmentations sequentially with random severity adjustments
        img = image
        for aug in selected_augs:
            # Adjust severity randomly for each augmentation
            aug.severity = self.severity * random.uniform(0.8, 1.2)
            img = aug(img)
            
        return img