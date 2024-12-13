import numpy as np
from PIL import Image, ImageOps
from .base import BaseAugmentation
import random


class LocalDeformationAdvanced(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply advanced local deformations"""
        img_np = np.array(image)
        height, width = img_np.shape[:2]
        
        # Generate multiple deformation regions
        num_regions = random.randint(2, 4)
        for _ in range(num_regions):
            # Random region selection
            region_width = random.randint(width // 8, width // 4)
            region_height = random.randint(height // 8, height // 4)
            x = random.randint(0, width - region_width)
            y = random.randint(0, height - region_height)
            
            # Apply random deformation type
            deform_type = random.choice(['twist', 'stretch', 'compress'])
            
            if deform_type == 'twist':
                img_np = self._apply_twist(img_np, x, y, region_width, region_height)
            elif deform_type == 'stretch':
                img_np = self._apply_stretch(img_np, x, y, region_width, region_height)
            else:  # compress
                img_np = self._apply_compress(img_np, x, y, region_width, region_height)
                
        return Image.fromarray(img_np)
    
    def _apply_twist(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Apply twisting effect to a region"""
        region = Image.fromarray(img[y:y+h, x:x+w])
        angle = self.severity * 30
        region = region.rotate(angle, resample=Image.BILINEAR, center=(w // 2, h // 2))
        
        img[y:y+h, x:x+w] = np.array(region)
        return img
    
    def _apply_stretch(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Apply stretching effect to a region"""
        region = Image.fromarray(img[y:y+h, x:x+w])
        new_h = int(h * (1 + self.severity * 0.5))
        
        # Resize region
        resized = region.resize((w, new_h), Image.BILINEAR)
        
        # Crop or pad to original size
        if new_h > h:
            resized = resized.crop((0, (new_h - h) // 2, w, (new_h + h) // 2))
        
        img[y:y+h, x:x+w] = np.array(resized)
        return img
    
    def _apply_compress(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Apply compression effect to a region"""
        region = Image.fromarray(img[y:y+h, x:x+w])
        new_w = int(w * (1 - self.severity * 0.3))
        
        # Resize region
        resized = region.resize((new_w, h), Image.BILINEAR)
        
        # Pad resized image to original size
        padded = ImageOps.pad(resized, (w, h), method=Image.BILINEAR)
        
        img[y:y+h, x:x+w] = np.array(padded)
        return img


class RandomRotate(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply random rotation to image"""
        angle = random.uniform(-30, 30)  # -30도에서 30도 사이 랜덤 회전
        return image.rotate(angle, resample=Image.BILINEAR, expand=False)
