import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from .base import BaseAugmentation

class FrequencyFiltering:
    def __init__(self, high_pass: bool = True, severity: float = 0.2):
        self.high_pass = high_pass
        self.severity = severity

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply high-pass or low-pass filtering to the image.
        """
        img_array = np.array(image, dtype=np.float32) / 255.0
        if self.high_pass:
            # High-pass filter (remove low frequencies)
            filtered = img_array - gaussian_filter(img_array, sigma=self.severity * 5)
        else:
            # Low-pass filter (retain low frequencies)
            filtered = gaussian_filter(img_array, sigma=self.severity * 5)
        filtered = np.clip(filtered * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(filtered)
