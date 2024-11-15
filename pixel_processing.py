from PIL import Image
import numpy as np

def process_specular(input_path, output_path):
    image = Image.open(input_path)
    image = image.convert('RGBA')
    width, height = image.size
    
    data = np.array(image)

    for y in range(height):
        for x in range(width):
            r, g, b, a = data[y, x]

            # If pixel isn't black, set to white, else transparent
            if r != 0 and g != 0 and b != 0:
                data[y, x] = [255, 255, 255, 255]
            else:
                data[y, x] = [0, 0, 0, 0]

    manipulated_image = Image.fromarray(data)
    manipulated_image.save(output_path)

process_specular('test_specular.png', 'test_image_edit.png')
