import numpy as np
import cv2
from PIL import Image
from .base import BaseAugmentation

class GaussianNoise(BaseAugmentation):
    def __init__(self, severity: float):
        super().__init__(severity)
        self.rng = np.random.RandomState(42)
        
    def __call__(self, image: Image.Image) -> Image.Image:
        try:
            img_np = np.array(image).astype(np.float32)
            
            noise_level = max(1e-3, self.severity * 25.0)
            noise = self.rng.normal(0, noise_level, img_np.shape)
            
            noisy_img = img_np + noise
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            
            return Image.fromarray(noisy_img)
        
        except Exception as e:
            print(f"Error in GaussianNoise: {str(e)}")
            return image

class TextureCorruption(BaseAugmentation):
    def __init__(self, severity: float):
        super().__init__(severity)
        self.rng = np.random.RandomState(42)
        
    def __call__(self, image: Image.Image) -> Image.Image:
        try:
            img = np.array(image)
            corruption_type = self.rng.choice(['blur', 'noise', 'salt_pepper'])
            
            if corruption_type == 'blur':
                return self._apply_blur(img)
            elif corruption_type == 'noise':
                return self._apply_noise(img)
            else:
                return self._apply_salt_pepper(img)
                
        except Exception as e:
            print(f"Error in TextureCorruption: {str(e)}")
            return image
            
    def _apply_blur(self, img):
        kernel_size = self.rng.choice([3, 5, 7])
        return Image.fromarray(cv2.GaussianBlur(img, (kernel_size, kernel_size), 0))
        
    def _apply_noise(self, img):
        noise = self.rng.normal(0, 25, img.shape)
        noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
        
    def _apply_salt_pepper(self, img):
        prob = self.rng.uniform(0.01, 0.05)
        thresh = 1 - prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = self.rng.rand()
                if rdn < prob:
                    img[i][j] = 0
                elif rdn > thresh:
                    img[i][j] = 255
        return Image.fromarray(img)