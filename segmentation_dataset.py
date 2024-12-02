import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np


# Taken from pixel_processing.py
# CLASSES = [SHADOW, CAST_SHADOW, MIDTONE, HIGHLIGHT, BACKROUND]
greyColours = [40, 80, 125, 255, 200]


# Define a custom Dataset to load train and test images
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, labelTransform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.labelTransform = labelTransform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load input image (grayscale)
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('L')

        # Construct corresponding label image name
        label_name = img_name.replace('.png', '_output.png')
        label_path = os.path.join(self.label_dir, label_name)
        label = Image.open(label_path).convert('L')
        label = np.array(label)

        # Normalize label to integers (make sure they're within [0, 4])
        for colour in greyColours:
            label[label == colour] = greyColours.index(colour)
        
        # One-hot encode the label
        label = np.eye(5)[label]
        label = label.transpose((2, 0, 1))

        image = np.array(image)
        image = image / 255.0

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        image = image.float()

        label_tensor = torch.from_numpy(label).float()

        return image, label_tensor
