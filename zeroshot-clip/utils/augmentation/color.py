from PIL import Image, ImageDraw, ImageEnhance
import random
import numpy as np
from .base import BaseAugmentation


class ColorDistortion(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        try:
            # Saturation adjustment
            saturation = ImageEnhance.Color(image)
            image = saturation.enhance(self.severity * 2)
            
            # Brightness adjustment
            brightness = ImageEnhance.Brightness(image)
            image = brightness.enhance(self.severity * 1.5)
            
            # Contrast adjustment
            contrast = ImageEnhance.Contrast(image)
            image = contrast.enhance(self.severity * 1.5)
            
            return image
        except Exception as e:
            print(f"Error in ColorDistortion: {str(e)}")
            return image

class RedDotAnomaly(BaseAugmentation):
    def __init__(self, severity: float):
        super().__init__(severity)
        self.rng = random.Random(42)
        
    def __call__(self, image: Image.Image) -> Image.Image:
        try:
            img = image.copy()
            draw = ImageDraw.Draw(img, 'RGBA')
            width, height = img.size
            
            if width < 4 or height < 4:
                return image
            
            num_dots = self.rng.randint(1, 3)
            for _ in range(num_dots):
                dot_size = self.rng.randint(2, min(6, min(width, height) // 4))
                x = self.rng.randint(dot_size, max(dot_size + 1, width - dot_size))
                y = self.rng.randint(dot_size, max(dot_size + 1, height - dot_size))
                
                red_color = (255, 0, 0, self.rng.randint(150, 255))
                draw.ellipse(
                    [(x-dot_size, y-dot_size), (x+dot_size, y+dot_size)],
                    fill=red_color
                )
            
            return img
            
        except Exception as e:
            print(f"Error in RedDotAnomaly: {str(e)}")
            return image

class PatternInjection(BaseAugmentation):
    def __init__(self, severity: float):
        super().__init__(severity)
        self.rng = np.random.RandomState(42)
        
    def __call__(self, image: Image.Image) -> Image.Image:
        try:
            img = image.copy()
            draw = ImageDraw.Draw(img, 'RGBA')
            width, height = img.size
            
            pattern_type = self.rng.choice(['dots', 'scratches', 'stains'])
            
            if pattern_type == 'dots':
                return self._apply_dots(draw, width, height)
            elif pattern_type == 'scratches':
                return self._apply_scratches(draw, width, height)
            else:
                return self._apply_stains(draw, width, height)
                
        except Exception as e:
            print(f"Error in PatternInjection: {str(e)}")
            return image
            
    def _apply_dots(self, draw, width, height):
        num_dots = self.rng.randint(1, 4)
        for _ in range(num_dots):
            size = self.rng.randint(2, max(3, min(width, height) // 20))
            color = (
                self.rng.randint(200, 255),
                self.rng.randint(0, 50),
                self.rng.randint(0, 50),
                self.rng.randint(150, 255)
            )
            x = self.rng.randint(size, width - size)
            y = self.rng.randint(size, height - size)
            draw.ellipse([(x-size, y-size), (x+size, y+size)], fill=color)
        return draw._image
        
    def _apply_scratches(self, draw, width, height):
        num_scratches = self.rng.randint(1, 3)
        for _ in range(num_scratches):
            start_x = self.rng.randint(0, width)
            start_y = self.rng.randint(0, height)
            end_x = start_x + self.rng.randint(-width//4, width//4)
            end_y = start_y + self.rng.randint(-height//4, height//4)
            draw.line([(start_x, start_y), (end_x, end_y)],
                     fill=(150, 150, 150, 200),
                     width=self.rng.randint(1, 3))
        return draw._image
        
    def _apply_stains(self, draw, width, height):
        num_stains = self.rng.randint(1, 2)
        for _ in range(num_stains):
            x = self.rng.randint(0, width)
            y = self.rng.randint(0, height)
            size = self.rng.randint(5, max(6, min(width, height) // 10))
            color = (
                self.rng.randint(100, 200),
                self.rng.randint(100, 200),
                self.rng.randint(100, 200),
                self.rng.randint(100, 180)
            )
            draw.ellipse([(x-size, y-size), (x+size, y+size)], fill=color)
        return draw._image