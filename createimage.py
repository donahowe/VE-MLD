import os
from PIL import Image, ImageDraw
import random
from tqdm import tqdm

output_folder = '/data2/chengjunhao/data/LEAF/train/images1'
input_folder = '/data2/chengjunhao/data/LEAF/train/images'

os.makedirs(output_folder, exist_ok=True)
file_list = [filename for filename in os.listdir(input_folder) if filename.endswith('.jpg')]

with tqdm(total=len(file_list), desc="Processing images") as pbar:
  for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)

        # 0-origin
        output_path = os.path.join(output_folder, filename.replace('.jpg', '.jpg'))
        image.save(output_path)

        # 1-crop
        width, height = image.size
        rect_width = width // 2
        rect_height = height // 2
        x1 = random.randint(0, width - rect_width)
        y1 = random.randint(0, height - rect_height)
        x2 = x1 + rect_width
        y2 = y1 + rect_height
        cropped_image = image.crop((x1, y1, x2, y2))
        output_path = os.path.join(output_folder, filename.replace('.jpg', '1.jpg'))
        cropped_image.save(output_path)

        # 2-rotate
        rotated_image = image.rotate(45)
        output_path = os.path.join(output_folder, filename.replace('.jpg', '2.jpg'))
        rotated_image.save(output_path)

        # 3-mask
        width, height = image.size
        masked_image = image.copy()
        rect_width = width // 3
        rect_height = height // 3
        x1 = random.randint(0, width - rect_width)
        y1 = random.randint(0, height - rect_height)
        x2 = x1 + rect_width
        y2 = y1 + rect_height
        for x in range(x1, x2):
            for y in range(y1, y2):
                masked_image.putpixel((x, y), (0, 0, 0))  # 设置像素颜色为黑色
        output_path = os.path.join(output_folder, filename.replace('.jpg', '3.jpg'))
        masked_image.save(output_path)
        pbar.update(1)
print("done")