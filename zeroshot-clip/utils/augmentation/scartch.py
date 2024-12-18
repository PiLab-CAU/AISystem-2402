import numpy as np
from PIL import Image
from .base import BaseAugmentation

class RandomScratch(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        height, width = img_np.shape[:2]
        
        # 스크래치 개수 랜덤 설정
        n_scratches = np.random.randint(2, 5)
        
        for _ in range(n_scratches):
            # 스크래치 시작점
            start_x = np.random.randint(0, width)
            start_y = np.random.randint(0, height)
            
            # 스크래치 길이와 방향
            length = np.random.randint(width//4, width//2)
            angle = np.random.uniform(0, 2 * np.pi)
            
            # 스크래치 두께
            thickness = np.random.randint(1, 4)
            
            # 스크래치 색상 (밝은 회색 ~ 흰색)
            scratch_color = np.random.randint(200, 256)
            
            # 스크래치 그리기
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            
            from PIL import ImageDraw
            draw = ImageDraw.Draw(Image.fromarray(img_np))
            draw.line([(start_x, start_y), (end_x, end_y)], 
                     fill=(scratch_color,)*3, 
                     width=thickness)
        
        return Image.fromarray(img_np)