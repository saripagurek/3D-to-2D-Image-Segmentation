from PIL import Image
import os

input_directory = "UnprocessedImages/Shape1"
output_directory = "UnprocessedImages/240x"

target_width = 240
target_height = 135

if not os.path.exists(output_directory):
    os.makedirs(output_directory)


def resize_image(image_path, output_path, target_width, target_height):

    with Image.open(image_path) as img:

        img_aspect_ratio = img.width / img.height
        target_aspect_ratio = target_width / target_height
        
        # Resize the image while keeping it centered
        if img_aspect_ratio > target_aspect_ratio:
            # Image is wider than the target aspect ratio
            new_width = target_width
            new_height = int(target_width / img_aspect_ratio)
        else:
            # Image is taller than the target aspect ratio
            new_height = target_height
            new_width = int(target_height * img_aspect_ratio)
        
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Calculate the position to paste the image (centered)
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        img_cropped = img_resized.crop((left, top, right, bottom))
        img_cropped.save(output_path, 'PNG', optimize=True)


for filename in os.listdir(input_directory):
    if filename.lower().endswith('.png'):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        print(f'Resizing {filename}...')
        resize_image(input_path, output_path, target_width, target_height)

print('Image resizing complete')
