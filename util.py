from PIL import Image
import os

directory_path = "ProcessedImages/200x/Shape2"
processed_dir = "example.gif"

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

create_gif_from_pngs(directory_path, processed_dir, fps=30)
