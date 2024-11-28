from PIL import Image
import numpy as np
import os


directory_path = "UnprocessedImages/200x/Shape10"
processed_dir = "ProcessedImages/200x/Shape10"

# Treating these as hardcoded, if these change, need to change the SegmentationDataset:11
white = [255, 255, 255, 255]
trns = [0, 0, 0, 0]
midtone = [125, 125, 125, 255]
shadow = [80, 80, 80, 255]
cast_shadow = [40, 40, 40, 255]


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
                data[y, x] = cast_shadow

    manipulated_image = Image.fromarray(data)
    manipulated_image.save(output_path)


def process_illum(input_path, mat_path, output_path):
    image = Image.open(input_path)
    image = image.convert('RGBA')

    mat_image = Image.open(mat_path)
    mat_image = mat_image.convert('RGBA')
    width, height = image.size
    
    data = np.array(image)
    mat_data = np.array(mat_image)

    for y in range(height):
        for x in range(width):
            r, g, b, a = data[y, x]
            mat_r, mat_g, mat_b, mat_a = mat_data[y, x]

            if (mat_r == 0 and mat_g == 0 and mat_b == 0):
                data[y, x] = trns
            else:
                # If pixel is dark enough to be in shadow:
                if (r < 100 and g < 100 and b < 100):
                    data[y, x] = shadow
                else:
                    data[y, x] = trns

    manipulated_image = Image.fromarray(data)
    manipulated_image.save(output_path)


def combine_layers(output_path, *input_paths):
    base_image = None

    for input_path in input_paths:
        image = Image.open(input_path).convert('RGBA')

        if base_image is None:
            base_image = Image.new('RGBA', image.size, (200, 200, 200, 255))
        
        base_image = Image.alpha_composite(base_image, image)

    base_image.save(output_path)


def get_files(directory):
    matcolour_files = []
    illum_files = []
    shadow_files = []
    specular_files = []
    input_files = []

    try:
        for filename in os.listdir(directory):
            if "matcolor" in filename.lower():
                matcolour_files.append(filename)
            elif "illum" in filename.lower():
                illum_files.append(filename)
            elif "shadow" in filename.lower():
                shadow_files.append(filename)
            elif "specular" in filename.lower():
                specular_files.append(filename)
            elif "diffuse" not in filename.lower():
                input_files.append(filename)
    except FileNotFoundError:
        print(f"The directory '{directory}' was not found.")
    except PermissionError:
        print(f"Permission denied for accessing the directory '{directory}'.")

    for file in matcolour_files:
        file = file[:-4]
        input = "" + directory_path + "/" + file + ".png"
        output = "" + processed_dir + "/" + file + "_edit.png"
        process_matcolour(input, output)
        print(f'Processing {file}...')

    for file in illum_files:
        file = file[:-4]
        shape = file[:-9]
        num = file[12:]
        mat = "" + processed_dir + "/" + shape + "matcolor" + num + "_edit.png"
        input = "" + directory_path + "/" + file + ".png"
        output = "" + processed_dir + "/" + file + "_edit.png"
        process_illum(input, mat, output)
        print(f'Processing {file}...')

    for file in shadow_files:
        file = file[:-4]
        input = "" + directory_path + "/" + file + ".png"
        output = "" + processed_dir + "/" + file + "_edit.png"
        process_shadow(input, output)
        print(f'Processing {file}...')

    for file in specular_files:
        file = file[:-4]
        input = "" + directory_path + "/" + file + ".png"
        output = "" + processed_dir + "/" + file + "_edit.png"
        process_specular(input, output)
        print(f'Processing {file}...')

    for file in input_files:
        if "png" in file:
            name = file[:-4]
            shape = file[:-9]
            num = file[7:-4]
            output = "" + processed_dir + "/" + name + "_output.png"
            mat = "" + processed_dir + "/" + shape + "_" + "matcolor" + num + "_edit.png"
            illum = "" + processed_dir + "/" + shape + "_" + "illum" + num + "_edit.png"
            shadow = "" + processed_dir + "/" + shape + "_" + "shadow" + num + "_edit.png"
            spec = "" + processed_dir + "/" + shape + "_" + "specular" + num + "_edit.png"

            combine_layers(
                output,
                mat,
                illum,
                shadow,
                spec
            )
            print(f'Processing {file}...')



get_files(directory_path)



'''
process_specular('test_specular.png', 'test_specular_edit.png')
process_matcolour('test_matcolour.png', 'test_matcolour_edit.png')
process_shadow('test_shadow.png', 'test_shadow_edit.png')
process_illum('test_illum.png', 'test_matcolour_edit.png', 'test_illum_edit.png')

combine_layers(
    'final_output.png',
    'test_matcolour_edit.png',
    'test_illum_edit.png',
    'test_shadow_edit.png',
    'test_specular_edit.png'
)
'''
