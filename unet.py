import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import subprocess


def remove_ds_store_files():
    try:
        # Run the find command
        subprocess.run(
            ['find', './data', '-name', '.DS_Store', '-type', 'f', '-delete'],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while removing .DS_Store files: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


remove_ds_store_files()


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
            self, in_channels=1, out_channels=5, features=[64, 128, 256, 512],
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
        label = np.clip(label, 0, 4)
        image = np.array(image)
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


# Load full dataset
full_dataset = SegmentationDataset(image_dir="data/train_images", label_dir="data/train_results", transform=transform)

# Split the dataset into training & validation
train_size = int(0.6 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


# Define a custom loss function that combines CrossEntropyLoss and Edge-Aware Regularization
class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_edge=0.0001):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()  # Standard CrossEntropyLoss
        self.weight_ce = weight_ce  # The weight for the cross-entropy loss
        self.weight_edge = weight_edge  # The weight for the edge loss term

    def forward(self, outputs, labels):
        # Compute the standard cross-entropy loss
        ce_loss = self.cross_entropy_loss(outputs, labels)

        # Compute the edge regularization loss
        edge_loss = self.compute_edge_loss(outputs)

        # Combine the losses with weights
        total_loss = (self.weight_ce * ce_loss) + (self.weight_edge * edge_loss)
        return total_loss

    def compute_edge_loss(self, outputs):
        """
        Compute the edge loss based on the difference between adjacent pixels
        using absolute difference (or squared difference if needed).
        """
        # Convert outputs to probabilities (logits -> softmax)
        probs = torch.softmax(outputs, dim=1)  # Shape: [batch_size, num_classes, height, width]

        # Get the horizontal and vertical pixel differences
        diff_x = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])  # Horizontal difference
        diff_y = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])  # Vertical difference

        # Sum all differences in the image (across both directions)
        edge_loss = (diff_x.sum() + diff_y.sum()) / (probs.shape[2] * probs.shape[3])

        # Optionally, refine edge loss by focusing on regions near boundaries
        # You could add a mask or a more targeted loss calculation if needed

        return edge_loss



# Initialize the model, loss function, and optimizer
model = UNET(in_channels=1, out_channels=5)

# Define the loss function
# criterion = nn.CrossEntropyLoss()  # Original loss function (only CrossEntropyLoss)
criterion = CombinedLoss()  # Combined loss (with edge loss)

optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Move the model to the appropriate device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")



# Training and Validation loop
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    print("Training started")
    model.train()
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs} started at {time.strftime('%H:%M:%S', time.gmtime(start_time))}")

        # Training phase
        running_train_loss = 0.0
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

            running_train_loss += loss.item()

        # Store average train loss for this epoch
        train_loss = running_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device).long()

                # Forward pass
                outputs = model(inputs)

                # Compute loss (raw logits)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                # Convert raw logits to class predictions (only for visualization)
                predicted_classes = torch.argmax(outputs, dim=1)


        # Store average validation loss for this epoch
        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)


        end_time = time.time()
        print(f"Epoch {epoch+1} finished. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} at {time.strftime('%H:%M:%S', time.gmtime(end_time))}")

    return train_losses, val_losses


# Train and validate the model
train_losses, val_losses = train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)

# Plot the loss function over time (across epochs)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', label='Validation Loss')
plt.title('Training vs Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# Inference on test images
def predict(model, test_image_dir, output_dir):
    print("Prediction started")
    model.eval()
    
    for test_file in os.listdir(test_image_dir):
        if test_file.endswith('.png'):
            test_path = os.path.join(test_image_dir, test_file)
            image = Image.open(test_path).convert('L')

            # Convert image to numpy array and normalize
            image = np.array(image)
            image = image / 255.0
            
            # Convert the image to a PyTorch tensor
            image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimension

            # Add batch dimension
            image = image.to(device).float()

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
