import numpy as np
from PIL import Image
from .base import BaseAugmentation

class Rotate(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        # 각 구간의 길이가 90도이며 severity만큼 축소
        interval_length = 90 
        
        negative_start = -180
        negative_end = -180 + interval_length  # -90 when severity=1.0
        
        positive_start = 180 - interval_length
        positive_end = 180

        # 음수 범위 혹은 양수 범위를 랜덤하게 선택
        if np.random.rand() < 0.5:
            angle = np.random.uniform(negative_start, negative_end)
        else:
            angle = np.random.uniform(positive_start, positive_end)

        return image.rotate(angle, expand=True)
