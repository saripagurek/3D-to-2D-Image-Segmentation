from PIL import Image, ImageSequence
import os
import shutil
import subprocess


# Remove ds store files if on mac before running unet.py to avoid file confusion
def remove_ds_store_files():
    try:
        # Run the find command
        subprocess.run(
            ['find', './', '-name', '.DS_Store', '-type', 'f', '-delete'],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while removing .DS_Store files: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Stitch a directory of frames together, output as a gif file
def create_gif_from_pngs(input_dir, output_path, fps=30):
    duration = int(1000 / fps)
    
    png_files = sorted([f for f in os.listdir(input_dir) if (f.endswith('.png'))])
    png_files = png_files[:91]
    if not png_files:
        raise ValueError("No PNG images found in the specified directory.")
    
    images = [Image.open(os.path.join(input_dir, file)) for file in png_files]
    
    # Save as a GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF saved at {output_path}")


# Stitches two GIFs side by side and saves the resulting GIF
def stitch_gifs_side_by_side(gif1_path, gif2_path, output_path):

    gif1 = Image.open(gif1_path)
    gif2 = Image.open(gif2_path)

    # Check if resolutions match
    if gif1.size[1] != gif2.size[1]:
        raise ValueError("The heights of the two GIFs must match.")

    # Create a new blank image with the combined width
    combined_width = gif1.width + gif2.width
    combined_height = gif1.height

    # List to store stitched frames
    frames = []

    # Get frames for both GIFs and stitch them
    for frame1, frame2 in zip(ImageSequence.Iterator(gif1), ImageSequence.Iterator(gif2)):
        # Ensure both frames have the same size
        frame1 = frame1.resize((gif1.width, gif1.height))
        frame2 = frame2.resize((gif2.width, gif2.height))

        # Create a new image for the combined frame
        combined_frame = Image.new('RGBA', (combined_width, combined_height))

        # Paste both frames side by side
        combined_frame.paste(frame1, (0, 0))
        combined_frame.paste(frame2, (gif1.width, 0))

        # Append to frames list
        frames.append(combined_frame)

    # Save the stitched frames as a new GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=gif1.info.get('duration', 100),  # Use the duration of the first GIF
        loop=gif1.info.get('loop', 0)  # Use the loop setting of the first GIF
    )


stitch_gifs_side_by_side("Examples/Shape2_input.gif", "Examples/Shape2_output.gif", "Examples/Shape2_stitched.gif")


# Helper function to copy files to a directory
def copy_files(src_dirs, dst_dir, condition):

    for src_dir in src_dirs:
        for root, _, files in os.walk(src_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".png"):
                    if condition(file):
                        print(f"Copying file: {file_path}")
                        dst_file = os.path.join(dst_dir, file)
                        shutil.copy(file_path, dst_file)
                    else:
                        print(f"Skipped file (condition not met): {file_path}")
                else:
                    print(f"Skipped file (not .png): {file_path}")


# Util function to set up train images, train results, and test images into the correct ./data directories before unet.py training
def organize_files(list_of_train_images, list_of_train_labels, list_of_tests, output_dir="./data"):

    # Output directories
    train_images_dir = os.path.join(output_dir, "train_images")
    train_results_dir = os.path.join(output_dir, "train_results")
    test_images_dir = os.path.join(output_dir, "test_images")
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_results_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)

    copy_files(list_of_train_images, train_images_dir, lambda f: len(f) == 15)
    print("Completed copying train_images.\n")

    # Copy train results (files containing "output")
    copy_files(list_of_train_labels, train_results_dir, lambda f: "output" in f)
    print("Completed copying train_results.\n")

    # Copy test images (15 characters including extension)
    copy_files(list_of_tests, test_images_dir, lambda f: len(f) == 15)
    print("Completed copying test_images.\n")


# Change this list of train images if you wish to train on different data
'''
path_to_train_images = [
    "./UnprocessedImages/200x/Shape1",
    "./UnprocessedImages/200x/Shape2",
    "./UnprocessedImages/200x/Shape3"
]
'''

# This list must match the above list to ensure there is a correct label for each input image
'''
path_to_train_labels = [
    "./ProcessedImages/200x/Shape1",
    "./ProcessedImages/200x/Shape2",
    "./ProcessedImages/200x/Shape3"
]
path_to_tests = ["./UnprocessedImages/200x/Shape4"]
'''


#organize_files(path_to_train_images, path_to_train_labels, path_to_tests)


#remove_ds_store_files()


directory_path = "UnprocessedImages/200x/Shape6"
processed_dir = "example.gif"
# create_gif_from_pngs(directory_path, processed_dir, fps=30)

