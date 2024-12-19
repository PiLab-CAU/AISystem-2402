from PIL import Image, ImageDraw
import random
from utils.augmentation.anomaly_augmenter import AnomalyAugmenter




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

def augmentation_square(image):
    draw = ImageDraw.Draw(image)
    width, height = image.size 

    square_radius = 70

    x_min, x_max = width // 4, 3 * (width // 4)
    y_min, y_max = height // 4, 3 * (height // 4)

    center_x = random.randint(x_min + square_radius, x_max - square_radius)
    center_y = random.randint(y_min + square_radius, y_max - square_radius)

    left_up_point = (center_x - square_radius, center_y - square_radius)
    right_down_point = (center_x + square_radius, center_y + square_radius)

    draw.rectangle([left_up_point, right_down_point], fill='#696969', outline='#696969')
    return image


def augmentation_rectangle_light(image):

    draw = ImageDraw.Draw(image, "RGBA")  # Enable RGBA for transparency
    width, height = image.size 

    # Determine orientation randomly: 0 for horizontal, 1 for vertical
    orientation = random.choice([0, 1])

    if orientation == 0:  # Horizontal rectangle
        rect_width = random.randint(200, 250)
        rect_height = random.randint(30, 50)
    else:  # Vertical rectangle
        rect_width = random.randint(30, 50)
        rect_height = random.randint(200, 250)

    # Define random position within bounds
    x_min, x_max = width // 4, 3 * (width // 4)
    y_min, y_max = height // 4, 3 * (height // 4)

    left_up_point = (
        random.randint(x_min, x_max - rect_width),
        random.randint(y_min, y_max - rect_height)
    )
    right_down_point = (
        left_up_point[0] + rect_width,
        left_up_point[1] + rect_height
    )

    # Draw translucent white rectangle
    draw.rectangle(
        [left_up_point, right_down_point], 
        fill=(255, 255, 255, 180),  # RGBA color with transparency
        outline=(255, 255, 255, 180)  # Outline with the same transparency
    )

    return image


class anomaly_transform:
    def __init__(self):
        pass

    def transform(self, image, augmentation_type):
        img = image

        if augmentation_type == 0:
            img = img
        
        elif augmentation_type == 1:
            img = augmentation_circle(image)

        elif augmentation_type == 2:
            img = augmentation_square(image)

        elif augmentation_type == 3:
            img = augmentation_rectangle_light(image)

        return img

    







if __name__ == '__main__':
    img = Image.open('/home/work/YewonKim/AISystem-2402/train/Calculator/0001.jpg').convert('RGB')
    output_path = '/home/work/YewonKim/AISystem-2402/zeroshot-clip/ellipse.jpg'

    a_t = anomaly_transform()

    img = a_t.transform(img,3)
    #augmenter = AnomalyAugmenter(severity=0.4)
    #anomaly_image = augmenter.generate_anomaly(img)

    img.save(output_path)