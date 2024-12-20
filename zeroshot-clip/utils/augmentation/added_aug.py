import numpy as np
import cv2
from PIL import Image
from .base import BaseAugmentation

class DefectAugmentation(BaseAugmentation):
    def __init__(self, severity: float = 0.7, color: tuple = (255, 255, 255)):
        super().__init__(severity)
        self.color = color

class ScratchAugmentation(DefectAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        mask = self.get_object_mask(img_np)
        height, width = img_np.shape[:2]
        
        n_scratches = max(1, min(4, int(4 * self.severity)))
        
        for _ in range(n_scratches):
            start_point = self.get_random_point_on_object(mask)
            if start_point is None:
                continue
                
            start_x, start_y = start_point
            length = max(20, min(int(100 * self.severity), width//2))
            thickness = max(1, min(3, int(3 * self.severity)))
            angle = np.random.randint(0, 360)
            
            end_x = int(start_x + length * np.cos(np.radians(angle)))
            end_y = int(start_y + length * np.sin(np.radians(angle)))
            
            scratch_mask = np.zeros_like(mask)
            cv2.line(scratch_mask, (start_x, start_y), (end_x, end_y), 255, thickness)
            scratch_mask = cv2.bitwise_and(scratch_mask, mask)
            
            if len(img_np.shape) == 3:
                img_np[scratch_mask > 0] = self.color
            else:
                img_np[scratch_mask > 0] = self.color[0]
            
        return Image.fromarray(img_np)

class SpotAugmentation(DefectAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        mask = self.get_object_mask(img_np)
        height, width = img_np.shape[:2]
        
        n_spots = max(1, min(5, int(5 * self.severity)))
        
        for _ in range(n_spots):
            center_point = self.get_random_point_on_object(mask)
            if center_point is None:
                continue
                
            center_x, center_y = center_point
            radius = max(2, min(int(10 * self.severity), width//20))
            
            spot_mask = np.zeros_like(mask)
            cv2.circle(spot_mask, (center_x, center_y), radius, 255, -1)
            spot_mask = cv2.bitwise_and(spot_mask, mask)
            
            if len(img_np.shape) == 3:
                img_np[spot_mask > 0] = self.color
            else:
                img_np[spot_mask > 0] = self.color[0]
            
        return Image.fromarray(img_np)

class CrackAugmentation(DefectAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        mask = self.get_object_mask(img_np)
        height, width = img_np.shape[:2]
        
        start_point = self.get_random_point_on_object(mask)
        if start_point is None:
            return image
            
        start_x, start_y = start_point
        points = [(start_x, start_y)]
        n_branches = max(3, min(7, int(7 * self.severity)))
        
        crack_mask = np.zeros_like(mask)
        
        for _ in range(n_branches):
            if not points:
                break
                
            x, y = points[np.random.randint(0, len(points))]
            angle = np.random.randint(0, 360)
            length = max(20, min(int(50 * self.severity), width//10))
            
            end_x = int(x + length * np.cos(np.radians(angle)))
            end_y = int(y + length * np.sin(np.radians(angle)))
            
            end_x = max(0, min(end_x, width-1))
            end_y = max(0, min(end_y, height-1))
            
            cv2.line(crack_mask, (x, y), (end_x, end_y), 255, 2)
            points.append((end_x, end_y))
        
        crack_mask = cv2.bitwise_and(crack_mask, mask)
        if len(img_np.shape) == 3:
            img_np[crack_mask > 0] = self.color
        else:
            img_np[crack_mask > 0] = self.color[0]
        
        return Image.fromarray(img_np)
    
class ConcavityAugmentation(DefectAugmentation):
    def __init__(self, severity: float = 0.7, color: tuple = (100, 100, 100)):  # 움푹 파인 부분의 음영을 표현하기 위한 어두운 색상
        super().__init__(severity)
        self.color = color
        
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        mask = self.get_object_mask(img_np)
        height, width = img_np.shape[:2]
        
        # 움푹 파인 부분 생성 (1-2개)
        n_concavities = np.random.randint(1, 3)
        
        for _ in range(n_concavities):
            # 객체 위의 랜덤한 점 선택
            center_point = self.get_random_point_on_object(mask)
            if center_point is None:
                continue
                
            center_x, center_y = center_point
            
            # 찌그러짐/파임의 크기 설정
            major_axis = max(10, min(int(40 * self.severity), width//8))
            minor_axis = max(5, min(int(20 * self.severity), width//16))
            angle = np.random.randint(0, 360)
            
            # 타원형 마스크 생성
            concavity_mask = np.zeros_like(mask)
            cv2.ellipse(concavity_mask, 
                       (center_x, center_y),
                       (major_axis, minor_axis),
                       angle, 0, 360, 255, -1)
            
            # 객체 영역과 교차
            concavity_mask = cv2.bitwise_and(concavity_mask, mask)
            
            # 그라데이션 효과 생성
            if len(img_np.shape) == 3:
                # 원본 이미지의 색상값을 기준으로 그라데이션 생성
                original_color = img_np[center_y, center_x].astype(float)
                darker_color = np.clip(original_color * 0.7, 0, 255).astype(np.uint8)  # 30% 어둡게
                
                # 그라데이션 적용
                y_coords, x_coords = np.where(concavity_mask > 0)
                for y, x in zip(y_coords, x_coords):
                    # 중심으로부터의 거리에 따른 그라데이션
                    distance = np.sqrt((y-center_y)**2 + (x-center_x)**2)
                    ratio = distance / major_axis
                    color = original_color * ratio + darker_color * (1-ratio)
                    img_np[y, x] = color.astype(np.uint8)
            else:
                # 그레이스케일 이미지의 경우
                img_np[concavity_mask > 0] = int(self.color[0] * 0.7)
                
        return Image.fromarray(img_np)