from PIL import Image, ImageDraw
import numpy as np
from .base import BaseAugmentation

class RedDotAnomaly(BaseAugmentation):
    def __init__(self, severity: float = 0.7, min_dots: int = 1, max_dots: int = 5):
        super().__init__(severity)
        self.min_dots = min_dots
        self.max_dots = max_dots
        
    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        n_dots = np.random.randint(self.min_dots, self.max_dots + 1)
        
        min_dim = min(image.size)
        dot_radius = int(min_dim * 0.01 * self.severity)  

        for _ in range(n_dots):
            x = np.random.randint(dot_radius, image.size[0] - dot_radius)
            y = np.random.randint(dot_radius, image.size[1] - dot_radius)
            
            alpha = int(255 * self.severity) 
            draw.ellipse(
                [(x - dot_radius, y - dot_radius), 
                 (x + dot_radius, y + dot_radius)],
                fill=(255, 0, 0, alpha) 
            )
        
        return Image.alpha_composite(image, overlay).convert('RGB')