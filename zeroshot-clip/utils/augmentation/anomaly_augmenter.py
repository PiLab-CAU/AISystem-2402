import numpy as np
from PIL import Image, ImageDraw
from typing import List
import random
from .base import BaseAugmentation
from .noise import GaussianNoise
from .geometric import LocalDeformation
from .color import ColorDistortion

class RandomDeletion(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        height, width = img_np.shape[:2]
        
        num_patches = np.random.randint(1, 4)
        for _ in range(num_patches):
            x = np.random.randint(0, width - width//4)
            y = np.random.randint(0, height - height//4)
            patch_w = np.random.randint(width//8, width//4)
            patch_h = np.random.randint(height//8, height//4)
            img_np[y:y+patch_h, x:x+patch_w] = 0
            
        return Image.fromarray(img_np)
    
def augmentation_circle(image):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    circle_radius = random.choice([50,70])

    x_min, x_max = width // 4, 3 * (width // 4)
    y_min, y_max = height // 4, 3 * (height // 4)

    center_x = random.randint(x_min + circle_radius, x_max - circle_radius)
    center_y = random.randint(y_min + circle_radius, y_max - circle_radius)

    left_up_point = (center_x - circle_radius, center_y - circle_radius)
    right_down_point = (center_x + circle_radius, center_y + circle_radius)

    # color = random.choice(['red', 'black'])

    draw.ellipse([left_up_point, right_down_point], fill='#CD5C5C', outline='#CD5C5C')
    

    return image

def augmentation_rectangle(image):
    draw = ImageDraw.Draw(image)
    width, height = image.size 

    circle_radius = 70

    x_min, x_max = width // 4, 3 * (width // 4)
    y_min, y_max = height // 4, 3 * (height // 4)

    center_x = random.randint(x_min + circle_radius, x_max - circle_radius)
    center_y = random.randint(y_min + circle_radius, y_max - circle_radius)

    left_up_point = (center_x - circle_radius, center_y - circle_radius)
    right_down_point = (center_x + circle_radius, center_y + circle_radius)

    draw.rectangle([left_up_point, right_down_point], fill='#696969', outline='#696969')
    return image

class AnomalyAugmenter:
    def __init__(self, severity: float = 0.7):
        self.augmentations: List[BaseAugmentation] = [
            random.choice([augmentation_circle, augmentation_rectangle]),
            GaussianNoise(severity),
            LocalDeformation(severity),
            ColorDistortion(severity),
            ]
    
    def generate_anomaly(self, image: Image.Image) -> Image.Image:
        # Generate anomaly images by combining multiple augmentations
        num_augs = 2
        selected_augs = np.random.choice(self.augmentations, num_augs, replace=False)
        
        img = image
        for aug in selected_augs:
            img = aug(img)
            
        return img
