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
        # 핵심 augmentation 세트 정의
        core_deformations = [
            LocalDeformationAdvanced(self.severity * 0.8),  # 부분적 변형
            TextureDeformation(self.severity * 0.6)         # 표면 질감 변화
        ]
        
        # 추가 효과 세트 정의
        additional_effects = [
            GaussianNoise(self.severity * 0.4),       # 미세한 노이즈
            RandomErase(self.severity * 0.5),         # 부분 손실
            AdvancedColorDistortion(self.severity * 0.3)  # 미묘한 색상 변화
        ]
        
        # 항상 core deformation 적용
        img = image
        for aug in core_deformations:
            img = aug(img)
        
        # 추가 효과 중 하나만 선택 적용
        selected_effect = random.choice(additional_effects)
        img = selected_effect(img)
        
        return img
    