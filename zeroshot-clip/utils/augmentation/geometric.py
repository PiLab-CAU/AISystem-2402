import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import map_coordinates
from .base import BaseAugmentation

class LocalDeformation(BaseAugmentation):
    def __init__(self, severity: float):
        super().__init__(severity)
        self.rng = np.random.RandomState(42)
        
    def __call__(self, image: Image.Image) -> Image.Image:
        try:
            img_np = np.array(image)
            height, width = img_np.shape[:2]
            
            if height < 4 or width < 4:
                return image
            
            # 변형 강도 설정
            strength = self.rng.uniform(5, 15)
            
            # 변형 영역 설정
            region_size = self.rng.randint(
                min(width, height) // 6,
                min(width, height) // 3
            )
            
            # 변형 중심점
            center_x = self.rng.randint(region_size, width - region_size)
            center_y = self.rng.randint(region_size, height - region_size)
            
            # 변형 맵 생성
            y, x = np.mgrid[:height, :width]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # 변형 마스크 생성
            mask = dist_from_center < region_size
            
            # 변형이 적용될 픽셀 위치 계산
            angle = self.rng.uniform(0, 2*np.pi)
            displacement = np.maximum(0, (1 - dist_from_center/region_size)) * strength
            
            # 새로운 좌표 계산
            x_new = x.astype(np.float32)
            y_new = y.astype(np.float32)
            x_new[mask] += np.cos(angle) * displacement[mask]
            y_new[mask] += np.sin(angle) * displacement[mask]
            
            # 범위 제한
            x_new = np.clip(x_new, 0, width-1)
            y_new = np.clip(y_new, 0, height-1)
            
            # 보간법을 사용한 이미지 변형
            deformed = np.zeros_like(img_np)
            for channel in range(img_np.shape[2]):
                deformed[..., channel] = map_coordinates(
                    img_np[..., channel],
                    [y_new, x_new],
                    order=1,
                    mode='reflect'
                )
            
            return Image.fromarray(deformed.astype(np.uint8))
            
        except Exception as e:
            print(f"Error in LocalDeformation: {str(e)}")
            return image
        
class RandomDeletion(BaseAugmentation):
    def __init__(self, severity: float):
        super().__init__(severity)
        self.rng = np.random.RandomState(42)
        
    def __call__(self, image: Image.Image) -> Image.Image:
        try:
            img_np = np.array(image)
            height, width = img_np.shape[:2]
            
            if height < 4 or width < 4:
                return image
            
            div_factor = self.rng.randint(4, 8)
            patch_w = max(width // div_factor, 1)
            patch_h = max(height // div_factor, 1)
            
            x = self.rng.randint(0, max(1, width - patch_w))
            y = self.rng.randint(0, max(1, height - patch_h))
            
            y_start = max(0, y-3)
            y_end = min(height, y+patch_h+3)
            x_start = max(0, x-3)
            x_end = min(width, x+patch_w+3)
            
            if y_end > y_start and x_end > x_start:
                surrounding = img_np[y_start:y_end, x_start:x_end]
                if surrounding.size > 0:
                    fill_color = np.mean(surrounding.reshape(-1, surrounding.shape[-1]), axis=0)
                    noise = self.rng.normal(0, 2, (patch_h, patch_w, 3))
                    fill_area = np.tile(fill_color, (patch_h, patch_w, 1))
                    fill_area = np.clip(fill_area + noise, 0, 255)
                    img_np[y:min(y+patch_h, height), x:min(x+patch_w, width)] = fill_area
            
            return Image.fromarray(img_np.astype(np.uint8))
            
        except Exception as e:
            print(f"Error in RandomDeletion: {str(e)}")
            return image

class RandomCuts(BaseAugmentation):
    def __init__(self, severity: float):
        super().__init__(severity)
        self.rng = np.random.RandomState(42)
    
    def __call__(self, image: Image.Image) -> Image.Image:
        try:
            img_np = np.array(image)
            height, width = img_np.shape[:2]
            
            cut_type = self.rng.choice(['rectangle', 'triangle'])
            if cut_type == 'rectangle':
                return self._apply_rectangle_cut(img_np, height, width)
            else:
                return self._apply_triangle_cut(img_np, height, width)
                
        except Exception as e:
            print(f"Error in RandomCuts: {str(e)}")
            return image
    
    def _apply_rectangle_cut(self, img_np, height, width):
        cut_width = self.rng.randint(width // 8, width // 4)
        cut_height = self.rng.randint(height // 8, height // 4)
        x = self.rng.randint(0, width - cut_width)
        y = self.rng.randint(0, height - cut_height)
        
        fill_color = img_np[max(0, y-5):min(height, y+5), 
                          max(0, x-5):min(width, x+5)].mean(axis=(0, 1))
        img_np[y:y+cut_height, x:x+cut_width] = fill_color
        return Image.fromarray(img_np)
    
    def _apply_triangle_cut(self, img_np, height, width):
        points = np.array([
            [self.rng.randint(0, width), self.rng.randint(0, height)],
            [self.rng.randint(0, width), self.rng.randint(0, height)],
            [self.rng.randint(0, width), self.rng.randint(0, height)]
        ])
        
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        
        fill_color = img_np[mask > 0].mean(axis=0)
        img_np[mask > 0] = fill_color
        return Image.fromarray(img_np)