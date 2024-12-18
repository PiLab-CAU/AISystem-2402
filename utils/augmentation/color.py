from PIL import Image, ImageEnhance
from .base import BaseAugmentation

class ColorDistortion(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        saturation = ImageEnhance.Color(image)
        image = saturation.enhance(self.severity * 2)
        
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(self.severity * 1.5)
        
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(self.severity * 1.5)
        
        return image
