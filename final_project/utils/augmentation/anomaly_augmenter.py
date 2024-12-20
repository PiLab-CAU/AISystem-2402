import numpy as np
from PIL import Image
from typing import List
from .base import BaseAugmentation
from .noise import GaussianNoise
from .geometric import LocalDeformation
from .color import ColorDistortion
import random
import math
import numpy as np
from PIL import Image, ImageDraw
from .base import BaseAugmentation
import cv2
from utils.seed_utils import set_global_seed

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

class ScratchAnomaly(BaseAugmentation):
    """긁힘/스크래치 효과를 시뮬레이션하는 증강"""
    def __call__(self, image: Image.Image) -> Image.Image:
        img = image.copy()
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # 스크래치 개수 랜덤 결정
        n_scratches = np.random.randint(1, 4)
        
        for _ in range(n_scratches):
            # 스크래치 시작점
            start_x = np.random.randint(0, width)
            start_y = np.random.randint(0, height)
            
            # 스크래치 길이와 각도
            length = np.random.randint(width//8, width//3)
            angle = np.random.uniform(0, 2 * np.pi)
            
            # 끝점 계산
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            
            # 스크래치 그리기 (약간의 투명도를 가진 회색)
            color = (128, 128, 128, int(255 * self.severity))
            draw.line([(start_x, start_y), (end_x, end_y)], 
                     fill=color, width=2)
            
        return img

class WaterDropAnomaly(BaseAugmentation):
    """물방울/얼룩 효과를 시뮬레이션하는 증강"""
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        height, width = img_np.shape[:2]
        
        # 물방울 개수 랜덤 결정
        n_drops = np.random.randint(1, 4)
        
        for _ in range(n_drops):
            # 물방울 중심점과 크기
            center_x = np.random.randint(0, width)
            center_y = np.random.randint(0, height)
            radius = np.random.randint(10, 30)
            
            # 물방울 마스크 생성
            y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
            mask = x*x + y*y <= radius*radius
            
            # 블러 효과 적용
            blurred = cv2.GaussianBlur(img_np, (25, 25), 0)
            
            # 물방울 효과 적용
            alpha = self.severity * np.exp(-(x*x + y*y)/(2*(radius/2)**2))
            alpha = np.dstack([alpha, alpha, alpha])
            img_np = np.where(mask[:,:,None], 
                            img_np * (1-alpha) + blurred * alpha, 
                            img_np)
            
        return Image.fromarray(img_np.astype(np.uint8))

class RustCorrosionAnomaly(BaseAugmentation):
    """녹/부식 효과를 시뮬레이션하는 증강"""
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        height, width = img_np.shape[:2]
        
        # 녹 색상 정의 (갈색 계열)
        rust_colors = [
            [139, 69, 19],   # 새들 브라운
            [160, 82, 45],   # 시에나
            [165, 42, 42],   # 브라운
            [128, 70, 27],   # 녹슨 갈색
        ]
        
        # 녹 마스크 생성
        rust_mask = np.zeros((height, width), dtype=np.float32)
        
        # 여러 개의 녹 패치 생성
        n_patches = np.random.randint(2, 5)
        for _ in range(n_patches):
            center_x = np.random.randint(0, width)
            center_y = np.random.randint(0, height)
            
            # 불규칙한 모양의 녹 패치
            for i in range(height):
                for j in range(width):
                    dist = np.sqrt((i-center_y)**2 + (j-center_x)**2)
                    noise = np.random.normal(0, 20)
                    radius = np.random.randint(20, 50)
                    if dist + noise < radius:
                        rust_mask[i,j] = max(rust_mask[i,j], 
                                           1 - dist/radius)
        
        # 녹 효과 적용
        rust_color = np.array(random.choice(rust_colors))
        for i in range(3):
            img_np[:,:,i] = img_np[:,:,i] * (1 - rust_mask * self.severity) + \
                           rust_color[i] * rust_mask * self.severity
            
        return Image.fromarray(img_np.astype(np.uint8))

class CrackPatternAnomaly(BaseAugmentation):
    """균열 패턴을 시뮬레이션하는 증강"""
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        height, width = img_np.shape[:2]
        
        # 균열 시작점
        start_x = np.random.randint(width // 4, 3 * width // 4)
        start_y = np.random.randint(height // 4, 3 * height // 4)
        
        # 균열 패턴 생성
        mask = np.ones_like(img_np)
        points = [(start_x, start_y)]
        
        # 프랙탈 같은 균열 패턴 생성
        def generate_crack(x, y, angle, length, depth):
            if depth <= 0 or length < 5:
                return
                
            end_x = int(x + length * np.cos(angle))
            end_y = int(y + length * np.sin(angle))
            
            # 경계 체크
            end_x = np.clip(end_x, 0, width-1)
            end_y = np.clip(end_y, 0, height-1)
            
            # 균열 선 그리기
            cv2.line(mask, (x, y), (end_x, end_y), (0,0,0), 2)
            
            # 분기 생성
            if np.random.random() < 0.7:  # 분기 확률
                new_length = length * 0.7
                angle_offset = np.random.uniform(-0.5, 0.5)
                generate_crack(end_x, end_y, 
                             angle + angle_offset, 
                             new_length, depth-1)
                generate_crack(end_x, end_y, 
                             angle - angle_offset, 
                             new_length, depth-1)
        
        # 초기 균열 생성
        initial_length = np.random.randint(30, 70)
        initial_angle = np.random.uniform(0, 2 * np.pi)
        generate_crack(start_x, start_y, initial_angle, initial_length, 3)
        
        # 균열 효과 적용
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        img_np = img_np * (mask * (1 - self.severity) + self.severity)
        
        return Image.fromarray(img_np.astype(np.uint8))

class DirtStainAnomaly(BaseAugmentation):
    """먼지/얼룩 효과를 시뮬레이션하는 증강"""
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        height, width = img_np.shape[:2]
        
        # 얼룩 색상 (어두운 회색 ~ 갈색)
        stain_colors = [
            [64, 64, 64],    # 다크 그레이
            [101, 67, 33],   # 다크 브라운
            [82, 82, 82],    # 미디엄 그레이
        ]
        
        # 퍼킨 노이즈로 자연스러운 얼룩 패턴 생성
        def perlin_noise(x, y, seed=0):
            np.random.seed(seed)
            p = np.arange(256, dtype=int)
            np.random.shuffle(p)
            p = np.stack([p,p]).flatten()
            
            xi = x.astype(int)
            yi = y.astype(int)
            xf = x - xi
            yf = y - yi
            
            u = fade(xf)
            v = fade(yf)
            
            n00 = gradient(p[p[xi] + yi], xf, yf)
            n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
            n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
            n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
            
            x1 = lerp(n00, n10, u)
            x2 = lerp(n01, n11, u)
            
            return lerp(x1, x2, v)
        
        def fade(t): return 6 * t**5 - 15 * t**4 + 10 * t**3
        def lerp(a, b, x): return a + x * (b - a)
        def gradient(h, x, y):
            vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
            g = vectors[h % 4]
            return g[:,:,0] * x + g[:,:,1] * y
        
        # 얼룩 마스크 생성
        lin = np.linspace(0, 5, max(height, width), dtype=float)
        x, y = np.meshgrid(lin[:width], lin[:height])
        
        noise = perlin_noise(x, y, seed=np.random.randint(0, 100))
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
        # 임계값으로 얼룩 영역 설정
        threshold = np.random.uniform(0.5, 0.7)
        stain_mask = noise > threshold
        
        # 얼룩 효과 적용
        stain_color = np.array(random.choice(stain_colors))
        for i in range(3):
            img_np[:,:,i] = np.where(
                stain_mask,
                img_np[:,:,i] * (1 - self.severity) + stain_color[i] * self.severity,
                img_np[:,:,i]
            )
        
        return Image.fromarray(img_np.astype(np.uint8))

# 기존 AnomalyAugmenter 클래스 업데이트
class EnhancedAnomalyAugmenter:
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
            MissingPartAnomaly(severity),
            ScratchAnomaly(severity),
            WaterDropAnomaly(severity),
            RustCorrosionAnomaly(severity),
            CrackPatternAnomaly(severity),
            DirtStainAnomaly(severity)
        ]
        
        # 증강 조합 확률
        self.combination_prob = 0.3
    
    def generate_anomaly(self, image: Image.Image) -> Image.Image:
        """이상 패턴 생성"""
        img = image.copy()
        
        # 기본 증강 선택
        if np.random.random() < 0.5:
            base_aug = np.random.choice(self.base_augmentations)
            img = base_aug(img)
        
        # 도메인 특화 증강 적용
        n_augmentations = np.random.randint(1, 3)  # 1-2개의 증강 적용
        selected_augs = np.random.choice(
            self.domain_augmentations, 
            size=n_augmentations, 
            replace=False
        )
        
        for aug in selected_augs:
            img = aug(img)
        
        return img
