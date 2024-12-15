import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from .base import BaseAugmentation
import random

class GaussianNoise(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image).astype(np.float32)
        noise = np.random.normal(0, self.severity * 75, img_np.shape)
        noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

class TextureDeformation(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        img = image
        operations = [
            self._apply_noise,
            self._apply_blur,
            self._apply_sharpen,
            self._apply_emboss,
            self._apply_edge_enhance,
            self._apply_contour,
            self._apply_smooth  # 새로운 operation
        ]
        
        num_ops = random.randint(2, 4)
        selected_ops = random.sample(operations, num_ops)
        
        for op in selected_ops:
            img = op(img)
            
        return img
    
    def _apply_noise(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        noise_types = ['gaussian', 'speckle', 'salt_pepper', 'poisson']
        noise_type = random.choice(noise_types)
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, self.severity * 35, img_np.shape)
            noisy = img_np + noise
        elif noise_type == 'speckle':
            noise = np.random.normal(0, self.severity * 0.25, img_np.shape)
            noisy = img_np * (1 + noise)
        elif noise_type == 'poisson':
            noise = np.random.poisson(img_np * self.severity * 0.1) - img_np * self.severity * 0.1
            noisy = img_np + noise
        else:  # salt_pepper
            noisy = img_np.copy()
            pepper = random.uniform(0, self.severity * 0.15)
            salt = random.uniform(0, self.severity * 0.15)
            
            mask = np.random.random(img_np.shape[:2]) < pepper
            noisy[mask] = 0
            
            mask = np.random.random(img_np.shape[:2]) < salt
            noisy[mask] = 255
            
        return Image.fromarray(np.uint8(np.clip(noisy, 0, 255)))
    
    def _apply_blur(self, image: Image.Image) -> Image.Image:
        blur_types = ['gaussian', 'box', 'motion', 'composite']  # radial 대신 composite
        blur_type = random.choice(blur_types)
        
        if blur_type == 'gaussian':
            radius = self.severity * random.uniform(2, 4)
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        elif blur_type == 'box':
            radius = self.severity * random.uniform(2, 4)
            return image.filter(ImageFilter.BoxBlur(radius=radius))
        elif blur_type == 'composite':
            # Multiple blur combination
            img = image
            img = img.filter(ImageFilter.GaussianBlur(radius=self.severity))
            img = img.filter(ImageFilter.BLUR)
            return img
        else:  # motion blur simulation using directional blur
            img = image
            for _ in range(int(self.severity * 3)):
                img = img.filter(ImageFilter.BLUR)
            return img
    
    def _apply_sharpen(self, image: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Sharpness(image)
        factor = 1 + self.severity * random.uniform(2, 4)
        return enhancer.enhance(factor)
    
    def _apply_emboss(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.EMBOSS)
    
    def _apply_edge_enhance(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            return image.filter(ImageFilter.EDGE_ENHANCE)
        else:
            return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    def _apply_contour(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.CONTOUR)
    
    def _apply_smooth(self, image: Image.Image) -> Image.Image:
        # Smooth와 Smooth More 필터 랜덤 적용
        if random.random() < 0.5:
            return image.filter(ImageFilter.SMOOTH)
        else:
            return image.filter(ImageFilter.SMOOTH_MORE)

class RandomErase(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        num_regions = random.randint(2, 4)
        for _ in range(num_regions):
            rh = random.randint(h//8, h//4)
            rw = random.randint(w//8, w//4)
            
            x = random.randint(0, w - rw)
            y = random.randint(0, h - rh)
            
            fill_type = random.choice(['zero', 'noise', 'color', 'pattern'])  # pattern 추가
            
            if fill_type == 'zero':
                img_np[y:y+rh, x:x+rw] = 0
            elif fill_type == 'noise':
                noise = np.random.randint(0, 255, (rh, rw, 3))
                img_np[y:y+rh, x:x+rw] = noise
            elif fill_type == 'pattern':
                # 체크보드 패턴 생성
                pattern = np.zeros((rh, rw, 3), dtype=np.uint8)
                square_size = 4
                for i in range(0, rh, square_size):
                    for j in range(0, rw, square_size):
                        if (i//square_size + j//square_size) % 2:
                            pattern[i:i+square_size, j:j+square_size] = 255
                img_np[y:y+rh, x:x+rw] = pattern
            else:  # color
                color = np.random.randint(0, 255, 3)
                img_np[y:y+rh, x:x+rw] = color
        
        return Image.fromarray(img_np)
    
class GlitchEffect(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply digital glitch-like effects"""
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        # Random channel shift
        if random.random() < 0.5:
            offset = int(w * self.severity * 0.1)
            for c in random.sample(range(3), random.randint(1, 2)):  # 1-2 channels
                img_np[:, :, c] = np.roll(img_np[:, :, c], random.randint(-offset, offset))
        
        # Random scan lines
        if random.random() < 0.5:
            num_lines = int(h * self.severity * 0.1)
            lines_idx = np.random.choice(h, num_lines, replace=False)
            img_np[lines_idx, :] = np.random.randint(0, 255, (num_lines, w, 3))
        
        return Image.fromarray(img_np)