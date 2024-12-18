import numpy as np
from PIL import Image
import random
from typing import List
from .base import BaseAugmentation
from .noise import GaussianNoise, TextureDeformation, RandomErase, GlitchEffect
from .geometric import LocalDeformationAdvanced, RandomRotate
from .color import AdvancedColorDistortion

class RedDotEffect(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        height, width = img_np.shape[:2]
        
        num_dots = random.randint(1, 3)
        
        for _ in range(num_dots):
            radius = int(min(width, height) * self.severity * 0.02)  # severity에 따른 크기 조절
            x = random.randint(radius, width - radius)
            y = random.randint(radius, height - radius)
            
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if i*i + j*j <= radius*radius:
                        if 0 <= y+i < height and 0 <= x+j < width:
                            img_np[y+i, x+j] = [255, 0, 0]
    
        return Image.fromarray(img_np)
    
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
        
        # 추가 효과 세트 정의 (ColorDistortion 제거)
        additional_effects = [
            GaussianNoise(self.severity * 0.4),       # 미세한 노이즈
            RandomErase(self.severity * 0.5),         # 부분 손실
            RedDotEffect(self.severity * 0.6)         # 빨간 점 효과
        ]
        
        img = image
        for aug in core_deformations:
            img = aug(img)
        
        selected_effect = random.choice(additional_effects)
        img = selected_effect(img)
        
        return img