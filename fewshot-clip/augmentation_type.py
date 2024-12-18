from PIL import Image, ImageDraw
import random


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

    color = random.choice(['red', 'black'])

    draw.ellipse([left_up_point, right_down_point], fill=color, outline=color)
    

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

    draw.rectangle([left_up_point, right_down_point], fill='black', outline='black')
    return image











if __name__ == '__main__':
    img = Image.open('/home/work/YewonKim/AISystem-2402/train/Calculator/0001.jpg').convert('RGB')
    output_path = '/home/work/YewonKim/AISystem-2402/fewshot-clip/ellipse.jpg'

    draw_circle = augmentation_circle(img)

    draw_circle.save(output_path)