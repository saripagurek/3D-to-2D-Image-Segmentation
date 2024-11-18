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
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# Define the DoubleConv class for U-Net
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# Define the U-Net model
class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=5, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


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
        label = Image.open(label_path).convert('L')
        label = np.array(label)

        # Normalize label to integers (make sure they're within [0, 4])
        label = np.clip(label, 0, 4)
        image = np.array(image)[:, :, :3]
        image = image / 255.0
        label = label.astype(np.long)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        image = image.float()
        label_tensor = torch.from_numpy(label).long()

        return image, label_tensor


# Define transformations (resize and convert to tensor)
transform = transforms.Compose([
    transforms.ToTensor(),
])


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
check_labels_in_batch(train_loader)


# Initialize the model, loss function, and optimizer
model = UNET(in_channels=3, out_channels=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Check if GPU is available and move model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Training loop with time stamps and loss tracking
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    print("Training started")
    model.train()
    
    epoch_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs} started at {time.strftime('%H:%M:%S', time.gmtime(start_time))}")

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # Ensure the input tensors are on the right device and of the right type
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Store average loss for this epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)

        end_time = time.time()
        print(f"Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f} at {time.strftime('%H:%M:%S', time.gmtime(end_time))}")

    return epoch_losses

# Train the model
epoch_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=15)

# Plot the loss function over time (across epochs)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
plt.title('Loss Function Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Inference on test images
def predict(model, test_image_dir, output_dir):
    print("Prediction started")
    model.eval()
    
    for test_file in os.listdir(test_image_dir):
        if test_file.endswith('.png'):
            test_path = os.path.join(test_image_dir, test_file)
            image = Image.open(test_path).convert('RGBA')

            # Convert image to numpy array and normalize
            image = np.array(image)[:, :, :3]
            image = image / 255.0
            
            # Convert the image to a PyTorch tensor
            image = torch.tensor(image).permute(2, 0, 1).float()

            # Add batch dimension
            image = image.unsqueeze(0).to(device).float()

            # Forward pass for prediction
            with torch.no_grad():
                output = model(image)

            # Get the class with the highest probability
            predicted_class = torch.argmax(output, dim=1).squeeze(0)  # Remove batch dimension
            predicted_image = predicted_class.cpu().numpy().astype(np.uint8)

            # Map class index to grayscale (5 shades)
            result_image = Image.fromarray(predicted_image * 51)
            result_image.save(os.path.join(output_dir, test_file.replace('.png', '_result.png')))
    
    print("Prediction complete")

# Predict on test images
predict(model, test_image_dir="data/test_images", output_dir="data/test_results")
