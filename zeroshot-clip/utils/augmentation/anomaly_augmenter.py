import numpy as np
from PIL import Image
import random
from typing import List
from .base import BaseAugmentation
from .noise import GaussianNoise, TextureDeformation, RandomErase, GlitchEffect
from .geometric import LocalDeformationAdvanced, RandomRotate
from .color import AdvancedColorDistortion

class AnomalyAugmenter:
    def __init__(self, severity: float = 0.65):  # 0.7에서 약간 낮춤
        self.augmentations = [
            AdvancedColorDistortion(severity * 1.1),    # 색상 변형은 미세하게
            TextureDeformation(severity),               # 텍스처는 기본 강도
            LocalDeformationAdvanced(severity * 1.2),   # 로컬 변형은 약간 강화
            RandomRotate(severity),                     # 회전은 기본 강도
            GaussianNoise(severity * 0.8),             # 노이즈는 오히려 낮춤
            RandomErase(severity * 0.9),               # 영역 제거도 약하게
            GlitchEffect(severity)                      # 글리치는 기본 강도
        ]
        self.severity = severity

    def generate_anomaly(self, image: Image.Image) -> Image.Image:
        # 더 적은 수의 augmentation 적용
        num_augs = random.randint(2, 3)  # 2~3개만 적용
        selected_augs = random.sample(self.augmentations, num_augs)
        
        img = image
        for aug in selected_augs:
            # 변동폭도 줄임
            aug.severity = self.severity * random.uniform(0.8, 1.3)
            img = aug(img)
            
        return img