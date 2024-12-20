import numpy as np
from PIL import Image
from typing import List
from .base import BaseAugmentation
from .noise import GaussianNoise
from .geometric import LocalDeformation
from .color import ColorDistortion
from .added_aug import ScratchAugmentation, SpotAugmentation, CrackAugmentation

class AnomalyAugmenter:
    def __init__(self, severity: float = 0.7):
        self.basic_augmentations: List[BaseAugmentation] = [
            GaussianNoise(severity),
            LocalDeformation(severity),
            ColorDistortion(severity),
        ]
        
        self.defect_augmentations: List[BaseAugmentation] = [
            ScratchAugmentation(severity=severity, color=(255, 255, 255)),  # 흰색 스크래치
            ScratchAugmentation(severity=severity, color=(128, 128, 128)),  # 회색 스크래치
            SpotAugmentation(severity=severity, color=(255, 0, 0)),         # 빨간 점
            CrackAugmentation(severity=severity, color=(0, 0, 0))          # 검은 크랙
        ]
    
    def generate_anomaly(self, image: Image.Image) -> Image.Image:
        img = image

        n_basic = np.random.randint(1, 3)
        selected_basic = np.random.choice(self.basic_augmentations, n_basic, replace=False)
        selected_defect = np.random.choice(self.defect_augmentations, 1)
        
        selected_augs = list(selected_basic) + list(selected_defect)
        np.random.shuffle(selected_augs)
        
        for aug in selected_augs:
            try:
                img = aug(img)
            except Exception as e:
                print(f"Error applying augmentation {aug.__class__.__name__}: {str(e)}")
                continue
                
        return img