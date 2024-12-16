import numpy as np
from PIL import Image
from typing import List
from .base import BaseAugmentation
from .noise import GaussianNoise
from .geometric import LocalDeformation
from .color import ColorDistortion

import numpy as np
from PIL import Image, ImageDraw
from .base import BaseAugmentation
import cv2

class RedDotAnomaly(BaseAugmentation):
    """작은 빨간 점 이상치를 시뮬레이션하는 증강"""
    def __call__(self, image: Image.Image) -> Image.Image:
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        # 이미지 크기에 따른 점 크기 조절
        width, height = img.size
        min_dot_size = int(min(width, height) * 0.005)  # 0.5% of image size
        max_dot_size = int(min(width, height) * 0.02)   # 2% of image size
        
        # 랜덤한 위치에 1-3개의 빨간 점 생성
        num_dots = np.random.randint(1, 4)
        for _ in range(num_dots):
            # 랜덤 위치 선택
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            # 랜덤 크기 선택
            dot_size = np.random.randint(min_dot_size, max_dot_size)
            
            # 빨간 점 그리기 (약간의 투명도 추가)
            red_color = (255, 0, 0, int(255 * self.severity))
            draw.ellipse([x, y, x + dot_size, y + dot_size], 
                        fill=red_color)
        
        return img

class BreakageAnomaly(BaseAugmentation):
    """물건이 부러지거나 망가진 경우를 시뮬레이션하는 증강"""
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        height, width = img_np.shape[:2]
        
        # 크랙 또는 파손 패턴 생성
        mask = np.ones_like(img_np)
        
        # 랜덤한 시작점 선택
        start_x = np.random.randint(width // 4, 3 * width // 4)
        start_y = np.random.randint(height // 4, 3 * height // 4)
        
        # 균열 패턴 생성
        num_cracks = np.random.randint(2, 5)
        for _ in range(num_cracks):
            points = [(start_x, start_y)]
            current_x, current_y = start_x, start_y
            
            # 균열 선 생성
            num_segments = np.random.randint(3, 8)
            for _ in range(num_segments):
                angle = np.random.uniform(0, 2 * np.pi)
                length = np.random.randint(10, 50)
                new_x = int(current_x + length * np.cos(angle))
                new_y = int(current_y + length * np.sin(angle))
                points.append((new_x, new_y))
                current_x, current_y = new_x, new_y
            
            # 균열 그리기
            points = np.array(points)
            cv2.polylines(mask, [points], False, (0, 0, 0), 
                         thickness=np.random.randint(1, 4))
            
        # 손상된 부분 처리
        damaged_img = img_np * (mask * (1 - self.severity) + self.severity)
        return Image.fromarray(damaged_img.astype(np.uint8))

class MissingPartAnomaly(BaseAugmentation):
    """부품이 없는 경우를 시뮬레이션하는 증강"""
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        height, width = img_np.shape[:2]
        
        # 랜덤한 위치와 크기로 부품 제거 영역 선택
        x = np.random.randint(0, width - width//4)
        y = np.random.randint(0, height - height//4)
        w = np.random.randint(width//8, width//4)
        h = np.random.randint(height//8, height//4)
        
        # 마스크 생성
        mask = np.ones_like(img_np)
        
        # 부품이 없는 효과 생성 방법 선택
        effect_type = np.random.choice(['remove', 'blur', 'background'])
        
        if effect_type == 'remove':
            # 완전히 제거 (검은색 또는 흰색)
            color = np.random.choice([0, 255])
            mask[y:y+h, x:x+w] = color
            
        elif effect_type == 'blur':
            # 해당 영역을 블러처리
            blurred = cv2.GaussianBlur(img_np[y:y+h, x:x+w], (25, 25), 0)
            img_np[y:y+h, x:x+w] = blurred
            
        else:  # background
            # 주변 배경색으로 채우기
            bg_color = np.median(img_np[y:y+h, x:x+w], axis=(0,1))
            noise = np.random.normal(0, 10, (h, w, 3))
            img_np[y:y+h, x:x+w] = np.clip(bg_color + noise, 0, 255)
        
        # 경계 부분 블렌딩
        kernel_size = int(min(w, h) * 0.1)
        kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
        
        return Image.fromarray(img_np.astype(np.uint8))

# 기존 AnomalyAugmenter 클래스 업데이트
class AnomalyAugmenter:
    def __init__(self, severity: float = 0.7):
        # 기본 증강
        self.base_augmentations: List[BaseAugmentation] = [
            GaussianNoise(severity),
            LocalDeformation(severity),
            ColorDistortion(severity)
        ]
        
        # 도메인 특화 증강
        self.domain_augmentations: List[BaseAugmentation] = [
            RedDotAnomaly(severity),
            BreakageAnomaly(severity),
            MissingPartAnomaly(severity)
        ]
    
    def generate_anomaly(self, image: Image.Image) -> Image.Image:
        # 도메인 특화 증강 중 하나를 선택
        domain_aug = np.random.choice(self.domain_augmentations)
        img = domain_aug(image)
        
        # 50% 확률로 기본 증강도 추가 적용
        if np.random.random() < 0.5:
            base_aug = np.random.choice(self.base_augmentations)
            img = base_aug(img)
            
        return img