from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
import cv2
from typing import Tuple

class BaseAugmentation(ABC):
    def __init__(self, severity: float = 0.7):
        self.severity = severity
    
    def get_object_mask(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5,5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, (255), -1)
            return mask
        return np.ones_like(gray) * 255

    def get_random_point_on_object(self, mask: np.ndarray) -> Tuple[int, int]:
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) == 0:
            return None
        idx = np.random.randint(0, len(y_coords))
        return x_coords[idx], y_coords[idx]

    @abstractmethod
    def __call__(self, image: Image.Image) -> Image.Image:
        pass