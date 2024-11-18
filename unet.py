import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import time
import matplotlib.pyplot as plt


# Define the UNet model with ReLU activation and CrossEntropy loss
class UNetWithReLUAndCrossEntropyLoss(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetWithReLUAndCrossEntropyLoss, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec1 = self.upconv_block(128, 64)
        
        # Final layer to output the number of grayscale classes (out_channels)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        # A standard convolution block with ReLU activation
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        # Upsampling block with transposed convolution and ReLU activation
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoding path
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        
        # Bottleneck
        bottleneck_out = self.bottleneck(enc4_out)
        
        # Decoding path
        dec4_out = self.dec4(bottleneck_out)
        dec3_out = self.dec3(self.crop_like(dec4_out, enc4_out) + enc4_out)  # Skip connection with cropping
        dec2_out = self.dec2(self.crop_like(dec3_out, enc3_out) + enc3_out)
        dec1_out = self.dec1(self.crop_like(dec2_out, enc2_out) + enc2_out)
        
        # Final layer to predict the grayscale classes
        final_out = self.final_conv(dec1_out)
        
        # Crop the output to match the input size
        return self.crop_like(final_out, x)

    def crop_like(self, x, target):
        # Crop the input `x` to match the size of `target`
        _, _, h, w = target.size()
        return x[:, :, :h, :w]


# Define a custom Dataset to load train and test images
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load input image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGBA')
        
        # Construct corresponding label image name
        label_name = img_name.replace('.png', '_output.png')
        label_path = os.path.join(self.label_dir, label_name)
        label = Image.open(label_path).convert('L')  # Convert to grayscale (L mode)
        label = np.array(label)

        # Check for out-of-bounds values in the labels
        unique_labels = np.unique(label)
        print(f"Unique label values for {img_name}: {unique_labels}")
        
        # Normalize label to integers (make sure they're within [0, 4])
        label = np.clip(label, 0, 4)
        
        # Normalize image to range [0, 1]
        image = np.array(image)[:, :, :3]  # Discard alpha channel and use RGB only
        image = image / 255.0
        label = label.astype(np.long)

        # Apply transformations (this already converts image to a tensor)
        if self.transform:
            image = self.transform(image)

        # Convert the image to float32 explicitly (to match model input)
        image = image.float()

        # Convert label to tensor manually as it's still in numpy format
        label_tensor = torch.from_numpy(label).long()

        return image, label_tensor


# Define transformations (resize and convert to tensor)
transform = transforms.Compose([
    transforms.ToTensor(),
])


# Visualization function to show input image and label map
def visualize_sample(input_image, label_map, idx):
    """Visualize one input image and its label map"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.set_title(f"Input Image {idx}")
    ax1.imshow(input_image.permute(1, 2, 0).cpu().numpy())  # Re-order to HWC for visualization
    
    ax2.set_title(f"Label Map {idx}")
    ax2.imshow(label_map.cpu().numpy(), cmap='tab20')  # 'tab20' colormap for distinct regions
    
    plt.show()


# Check unique labels in the first batch of the data loader
def check_labels_in_batch(data_loader):
    data_iter = iter(data_loader)
    inputs, labels = next(data_iter)
    
    for idx in range(inputs.shape[0]):
        unique_labels = torch.unique(labels[idx])
        print(f"Unique labels for sample {idx}: {unique_labels.cpu().numpy()}")
        
# Load training and validation datasets
train_dataset = SegmentationDataset(image_dir="data/train_images", label_dir="data/train_results", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Check the labels in the first batch
check_labels_in_batch(train_loader)


# Initialize the model, loss function, and optimizer
model = UNetWithReLUAndCrossEntropyLoss(in_channels=3, out_channels=5)
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Check if GPU is available and move model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    print("Training started")
    model.train()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} started")
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            print(f"Batch {i+1} started")

            # Check input and label sizes and types
            print(f"Input shape: {inputs.shape}, Label shape: {labels.shape}")
            print(f"Input dtype: {inputs.dtype}, Label dtype: {labels.dtype}")
            
            optimizer.zero_grad()

            # Ensure the input tensors are on the right device and of the right type
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()

            # Forward pass
            outputs = model(inputs)
            
            # Check output shape and dtype
            print(f"Output shape: {outputs.shape}, Output dtype: {outputs.dtype}")

            # Compute loss
            loss = criterion(outputs, labels)
            print(f"Batch {i+1} loss: {loss.item()}")

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print average loss for the epoch
        print(f"Epoch {epoch+1} finished. Average Loss: {running_loss/len(train_loader)}")


train_model(model, train_loader, criterion, optimizer, num_epochs=15)

# Inference on test images
def predict(model, test_image_dir, output_dir):
    print("Prediction started")
    model.eval()
    
    for test_file in os.listdir(test_image_dir):
        if test_file.endswith('.png'):
            test_path = os.path.join(test_image_dir, test_file)
            image = Image.open(test_path).convert('RGBA')

            # Convert image to numpy array and normalize
            image = np.array(image)[:, :, :3]  # Discard alpha channel and use RGB only
            image = image / 255.0  # Normalize image to range [0, 1]
            
            # Convert the image to a PyTorch tensor
            image = torch.tensor(image).permute(2, 0, 1).float()  # Convert shape from [H, W, C] to [C, H, W]

            # Add batch dimension (i.e., make it [1, C, H, W])
            image = image.unsqueeze(0).to(device).float()

            # Forward pass for prediction
            with torch.no_grad():
                output = model(image)

            # Get the class with the highest probability
            predicted_class = torch.argmax(output, dim=1).squeeze(0)  # Remove batch dimension
            predicted_image = predicted_class.cpu().numpy().astype(np.uint8)

            # Map class index to grayscale (5 shades)
            result_image = Image.fromarray(predicted_image * 51)  # Multiply by 51 to map to 5 grayscale shades
            result_image.save(os.path.join(output_dir, test_file.replace('.png', '_result.png')))
    
    print("Prediction complete")


# Predict on test images
predict(model, test_image_dir="data/test_images", output_dir="data/test_results")
