from PIL import Image
import numpy as np

white = [255, 255, 255, 255]
trns = [0, 0, 0, 0]
midtone = [125, 125, 125, 255]
shadow = [60, 60, 60, 255]


def process_specular(input_path, output_path):
    image = Image.open(input_path)
    image = image.convert('RGBA')
    width, height = image.size
    
    data = np.array(image)

    for y in range(height):
        for x in range(width):
            r, g, b, a = data[y, x]

            # If pixel isn't black, set to white, else transparent
            if not (r < 20 and g < 20 and b < 20):
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
            if (r < 20 and g < 20 and b < 20) or (r >= 210 and g >= 210 and b >= 210):
                # need a better way to remove the horizon line?
                data[y, x] = trns
            else:
                #print(data[y, x])
                data[y, x] = midtone

    manipulated_image = Image.fromarray(data)
    manipulated_image.save(output_path)


def process_shadow(input_path, output_path):
    image = Image.open(input_path)
    image = image.convert('RGBA')
    width, height = image.size
    
    data = np.array(image)

    for y in range(height):
        for x in range(width):
            r, g, b, a = data[y, x]

            # If pixel isn't white, set to shadow, else transparent
            if (r >= 210 and g >= 210 and b >= 210):
                data[y, x] = trns
            else:
                data[y, x] = shadow

    manipulated_image = Image.fromarray(data)
    manipulated_image.save(output_path)


def combine_layers(output_path, *input_paths):
    base_image = None

    for input_path in input_paths:
        image = Image.open(input_path).convert('RGBA')

        if base_image is None:
            base_image = Image.new('RGBA', image.size, (0, 0, 0, 255))
        
        base_image = Image.alpha_composite(base_image, image)

    base_image.save(output_path)



process_specular('test_specular.png', 'test_specular_edit.png')
process_matcolour('test_matcolour.png', 'test_matcolour_edit.png')
process_shadow('test_shadow.png', 'test_shadow_edit.png')

combine_layers(
    'final_output.png',
    'test_matcolour_edit.png',
    'test_shadow_edit.png',
    'test_specular_edit.png'
)

