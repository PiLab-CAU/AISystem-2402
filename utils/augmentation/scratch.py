import random
import numpy as np
from PIL import Image, ImageDraw
from .base import BaseAugmentation

class SpotScratchCrack(BaseAugmentation):
    def __call__(self, image: Image.Image) -> Image.Image:
        # severity에 따라 금이나 흠집의 개수나 크기를 조절
        width, height = image.size
        draw = ImageDraw.Draw(image)
        
        # 랜덤한 개수의 흠집을 추가 (개수는 severity 비례)
        num_marks = int(self.severity * 5)  # 예: severity 0.7 => 약 3~4개
        for _ in range(num_marks):
            # 랜덤한 시작점, 끝점 선택
            x1 = random.randint(0, width - 1)
            y1 = random.randint(0, height - 1)
            x2 = x1 + random.randint(-int(width * 0.1), int(width * 0.1))
            y2 = y1 + random.randint(-int(height * 0.1), int(height * 0.1))
            
            # 선 두께와 색 지정 (일반적으로 흰색 또는 어두운 색)
            line_width = int(self.severity * 3)  # 선 두께
            line_color = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)) if random.random() < 0.5 else (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            
            # 선 그리기 (금/흠집)
            draw.line((x1, y1, x2, y2), fill=line_color, width=line_width)

            # 가능하다면 점 형태의 결함도 추가
            if random.random() < 0.5:
                spot_size = int(self.severity * 10)
                spot_x = x1 + random.randint(-spot_size, spot_size)
                spot_y = y1 + random.randint(-spot_size, spot_size)
                # 원형 또는 사각형 형태의 반점
                draw.ellipse((spot_x, spot_y, spot_x+spot_size, spot_y+spot_size), fill=line_color)
        
        return image