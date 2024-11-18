from PIL import Image
import os

input_directory = "UnprocessedImages/Shape1"
output_directory = "UnprocessedImages/200x/Shape1"

target_width = 200
target_height = 200

if not os.path.exists(output_directory):
    os.makedirs(output_directory)


def rename(directory, replace, replace_with):
    try:
        for filename in os.listdir(directory):
            if replace in filename:
                new_filename = filename.replace(replace, replace_with)
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                os.rename(old_path, new_path)
                print(f'Renamed: {filename} -> {new_filename}')
    except Exception as e:
        print(f"An error occurred: {e}")

#rename(output_directory, "Shape_3", "Shape3")


def resize_image(image_path, output_path, target_width, target_height, zoom_factor=1.4):
    if zoom_factor < 1.0:
        raise ValueError("Zoom factor must be greater than or equal to 1.0.")
    
    with Image.open(image_path) as img:
        img_aspect_ratio = img.width / img.height
        target_aspect_ratio = target_width / target_height
        
        # Calculate cropping box for zoom
        zoomed_width = img.width / zoom_factor
        zoomed_height = img.height / zoom_factor
        left = (img.width - zoomed_width) / 2
        top = (img.height - zoomed_height) / 2
        right = left + zoomed_width
        bottom = top + zoomed_height

        img_zoomed = img.crop((left, top, right, bottom))
        
        # Resize the zoomed image to ensure it fills the target dimensions
        img_zoomed_aspect_ratio = zoomed_width / zoomed_height
        if img_zoomed_aspect_ratio > target_aspect_ratio:
            new_height = target_height
            new_width = int(target_height * img_zoomed_aspect_ratio)
        else:
            # Image is taller, fit width and crop height
            new_width = target_width
            new_height = int(target_width / img_zoomed_aspect_ratio)
        
        img_resized = img_zoomed.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Calculate cropping box to center the image
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
