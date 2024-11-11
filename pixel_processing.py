from PIL import Image
import numpy as np

def process_image(input_path, output_path):
    image = Image.open(input_path)
    image = image.convert('RGBA')
    width, height = image.size
    
    data = np.array(image)

    for y in range(height):
        for x in range(width):
            r, g, b, a = data[y, x]
            # Invert RGB values
            data[y, x] = [255 - r, 255 - g, 255 - b, a]
    
    manipulated_image = Image.fromarray(data)
    manipulated_image.save(output_path)

process_image('test_image.png', 'test_image_edit.png')
