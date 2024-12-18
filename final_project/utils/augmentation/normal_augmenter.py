from PIL import Image
import numpy as np
from typing import List
import torchvision.transforms.functional as TF

class NormalAugmenter:
    def __init__(self):
        self.rotation_angles = [90, 180, 270]  # 90도 단위로 회전
        self.flip_modes = ['horizontal', 'vertical']  # 수평, 수직 뒤집기
        
    def generate_augmented_images(self, image: Image.Image) -> List[Image.Image]:
        """
        주어진 이미지에 대해 회전과 뒤집기를 적용하여 증강된 이미지들을 생성

        Args:
            image: 원본 이미지

        Returns:
            List[Image.Image]: 증강된 이미지 리스트
        """
        augmented_images = []
        
        # 회전 적용
        for angle in self.rotation_angles:
            rotated_img = TF.rotate(image, angle)
            augmented_images.append(rotated_img)
            
            # 회전된 이미지에 대해 뒤집기 적용
            for flip_mode in self.flip_modes:
                if flip_mode == 'horizontal':
                    flipped_img = TF.hflip(rotated_img)
                else:
                    flipped_img = TF.vflip(rotated_img)
                augmented_images.append(flipped_img)
        
        # 원본 이미지에 대해 뒤집기만 적용
        for flip_mode in self.flip_modes:
            if flip_mode == 'horizontal':
                flipped_img = TF.hflip(image)
            else:
                flipped_img = TF.vflip(image)
            augmented_images.append(flipped_img)
            
        return augmented_images