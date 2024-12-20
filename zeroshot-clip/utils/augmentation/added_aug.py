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