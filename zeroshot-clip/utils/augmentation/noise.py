import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from .base import BaseAugmentation
import random

class GaussianNoise(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image).astype(np.float32)
        noise = np.random.normal(0, self.severity * 50, img_np.shape)
        noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    
class TextureDeformation(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply texture deformationss"""
        # Apply multiple texture modifications
        img = image
        
        # Random texture operations
        operations = [
            self._apply_noise,
            self._apply_blur,
            self._apply_sharpen,
            self._apply_emboss
        ]
        
        # Apply 2-3 random operations
        num_ops = random.randint(2, 3)
        selected_ops = random.sample(operations, num_ops)
        
        for op in selected_ops:
            img = op(img)
            
        return img
    
    def _apply_noise(self, image: Image.Image) -> Image.Image:
        """Add detailed noise patterns"""
        img_np = np.array(image)
        noise_types = ['gaussian', 'speckle', 'salt_pepper']
        noise_type = random.choice(noise_types)
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, self.severity * 25, img_np.shape)
            noisy = img_np + noise
        elif noise_type == 'speckle':
            noise = np.random.normal(0, self.severity * 0.15, img_np.shape)
            noisy = img_np * (1 + noise)
        else:  # salt_pepper
            noisy = img_np.copy()
            pepper = random.uniform(0, self.severity * 0.1)
            salt = random.uniform(0, self.severity * 0.1)
            
            # Pepper mode
            mask = np.random.random(img_np.shape[:2]) < pepper
            noisy[mask] = 0
            
            # Salt mode
            mask = np.random.random(img_np.shape[:2]) < salt
            noisy[mask] = 255
            
        return Image.fromarray(np.uint8(np.clip(noisy, 0, 255)))
    
    def _apply_blur(self, image: Image.Image) -> Image.Image:
        """Apply various blur effects"""
        blur_types = ['gaussian', 'box', 'motion']
        blur_type = random.choice(blur_types)
        
        if blur_type == 'gaussian':
            return image.filter(ImageFilter.GaussianBlur(radius=self.severity * 2))
        elif blur_type == 'box':
            return image.filter(ImageFilter.BoxBlur(radius=self.severity * 2))
        else:  # motion blur
            return image.filter(ImageFilter.BLUR)
    
    def _apply_sharpen(self, image: Image.Image) -> Image.Image:
        """Apply sharpening effect"""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(1 + self.severity * 2)
    
    def _apply_emboss(self, image: Image.Image) -> Image.Image:
        """Apply emboss effect"""
        return image.filter(ImageFilter.EMBOSS)