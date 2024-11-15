from PIL import Image
import numpy as np

white = [255, 255, 255, 255]
trns = [0, 0, 0, 0]
midtone = [125, 125, 125, 255]


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
                data[y, x] = white
            else:
                data[y, x] = trns

    manipulated_image = Image.fromarray(data)
    manipulated_image.save(output_path)


def process_matcolour(input_path, output_path):
    image = Image.open(input_path)
    image = image.convert('RGBA')
    width, height = image.size
    
    data = np.array(image)

    for y in range(height):
        for x in range(width):
            r, g, b, a = data[y, x]

            # If pixel isn't black or white, set to mid tone
            if (r < 20 and g < 20 and b < 20) or (r >= 100 and g >= 100 and b >= 100):
                # need a better way to remove the horizon line
                data[y, x] = trns
            else:
                #print(data[y, x])
                data[y, x] = midtone

    manipulated_image = Image.fromarray(data)
    manipulated_image.save(output_path)


process_specular('test_specular.png', 'test_specular_edit.png')
process_matcolour('test_matcolour.png', 'test_matcolour_edit.png')
