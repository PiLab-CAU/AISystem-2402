from PIL import Image, ImageEnhance
from .base import BaseAugmentation
import numpy as np
import random

class AdvancedColorDistortion(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply advanced color distortions"""
        try:
            # Convert to HSV using PIL's built-in functionality
            img = image.convert('HSV')
            
            # Split into channels
            h, s, v = img.split()
            
            # Apply random adjustments to each channel
            h_shift = int(random.uniform(-10, 10))  # Hue shift
            s_factor = random.uniform(0.5, 1.5)     # Saturation factor
            v_factor = random.uniform(0.5, 1.5)     # Value/brightness factor
            
            # Adjust hue (circular shift) with proper bounds checking
            h_data = np.array(h)
            h_data = np.clip((h_data.astype(np.int16) + h_shift) % 256, 0, 255).astype(np.uint8)
            h = Image.fromarray(h_data)
            
            # Adjust saturation and value
            s = ImageEnhance.Brightness(s).enhance(s_factor)
            v = ImageEnhance.Brightness(v).enhance(v_factor)
            
            # Merge channels back
            img = Image.merge('HSV', (h, s, v))
            
            # Convert back to RGB
            img = img.convert('RGB')
            
            # Apply additional color adjustments with more conservative factors
            enhancers = [
                (ImageEnhance.Contrast, random.uniform(0.9, 1.1)),
                (ImageEnhance.Color, random.uniform(0.9, 1.1))
            ]
            
            for enhancer_class, factor in enhancers:
                img = enhancer_class(img).enhance(factor)
                
            return img
            
        except Exception as e:
            print(f"Error in color distortion: {str(e)}")
            return image  # Return original image if transformation fails