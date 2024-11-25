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

num_epochs = 5
batch_size = 16


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
                # (First line for CUDA GPU, second for MAC GPU)
                #x = TF.resize(x, size=skip_connection.shape[2:])
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)


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


# Define a custom loss function that combines CrossEntropyLoss and Edge-Aware Regularization
class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=1.1, weight_edge=0.001, weight_consistency=0.1):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()  # Standard CrossEntropyLoss
        self.weight_ce = weight_ce  # The weight for the cross-entropy loss
        self.weight_edge = weight_edge  # The weight for the edge loss term
        self.weight_consistency = weight_consistency  # The weight for the consistency term

    def forward(self, outputs, labels):
        # Compute the standard cross-entropy loss
        ce_loss = self.cross_entropy_loss(outputs, labels)

        # Compute the edge regularization loss
        edge_loss = self.compute_edge_loss(outputs)

        # Compute the local consistency loss
        consistency_loss = self.compute_local_consistency_loss(outputs)

        # Combine the losses with weights
        total_loss = (self.weight_ce * ce_loss) + (self.weight_edge * edge_loss) + (self.weight_consistency * consistency_loss)
        return total_loss

    def compute_edge_loss(self, outputs):
        """
        Compute the edge loss based on the difference between adjacent pixels
        using absolute difference (or squared difference if needed).
        """
        # Convert outputs to probabilities (logits -> softmax)
        probs = torch.softmax(outputs, dim=1)  # Shape: [batch_size, num_classes, height, width]

        # Get the horizontal and vertical pixel differences
        diff_x = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
        diff_y = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])

        # Sum all differences in the image (across both directions)
        edge_loss = (diff_x.sum() + diff_y.sum()) / (probs.shape[2] * probs.shape[3])

        return edge_loss

    def compute_local_consistency_loss(self, outputs):
        """
        Compute the consistency loss to ensure no isolated pixels differ from their neighbors.
        For each pixel, we check its 4-connected neighbors (up, down, left, right).
        If all neighbors are the same class, but the pixel differs, apply a penalty.
        """
        # Convert logits to predicted class labels
        _, predicted_classes = torch.max(outputs, dim=1)  # Shape: [batch_size, height, width]

        # Create a zero tensor to accumulate the consistency penalty
        consistency_loss = 0.0
        batch_size, height, width = predicted_classes.shape

        # Loop over the image (avoid borders to prevent out-of-bound errors)
        for b in range(batch_size):
            for i in range(height): 
                for j in range(width):
                    # Get the central pixel class and its neighbors (top, bottom, left, right)
                    center = predicted_classes[b, i, j]
                    neighbors = []

                    # Check if the neighbors exist, adding them to the list
                    if i > 0:
                        neighbors.append(predicted_classes[b, i-1, j])
                    if i < height-1:
                        neighbors.append(predicted_classes[b, i+1, j])
                    if j > 0:
                        neighbors.append(predicted_classes[b, i, j-1])
                    if j < width-1:
                        neighbors.append(predicted_classes[b, i, j+1])

                    # Check if all neighbors are the same class and if the central pixel differs
                    if len(set(neighbors)) == 1 and center != neighbors[0]:
                        consistency_loss += 1

        # Normalize by the number of pixels (to keep the loss scale consistent)
        consistency_loss /= (batch_size * height * width)

        return consistency_loss


# Training and Validation loop
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=None):
    print("Training started")
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs} started at {time.strftime('%H:%M:%S', time.gmtime(start_time))}")

        # Training phase
        running_train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            print(f"Batch {i+1}/{len(train_loader)}", end='\r')
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

        average_time_taken = 0.0

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

        #get the time taken for the epoch
        time_taken = end_time - start_time

        print(f"Epoch {epoch+1} finished. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} at {time.strftime('%H:%M:%S', time.gmtime(end_time))} (Took {time_taken:.2f} seconds to run)")

        torch.save(model.state_dict(), "weights/weights_" + str(epoch) + ".pth")

    print("Training complete")

    average_time_taken = average_time_taken / num_epochs
    print(f"Average time taken per epoch: {average_time_taken:.2f} seconds")

    return train_losses, val_losses




# Inference on test images
def predict(model, test_image_dir, output_dir, device=None):
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



def main():

    remove_ds_store_files()


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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # Initialize the model, loss function, and optimizer
    model = UNET(in_channels=1, out_channels=5)

    # Define the loss function
    # criterion = nn.CrossEntropyLoss()  # Original loss function (only CrossEntropyLoss)
    criterion = CombinedLoss(weight_ce=1.1, weight_edge=0.001, weight_consistency=0.2)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Move the model to the appropriate device (First line for CUDA GPU, second for MAC GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")



    #Ask the user if they want to train the model or load weights
    train = input("Do you want to train the model? (y/n): ")
    if train == 'y':
        # Train and validate the model
        print("Training on ", len(train_loader), " batches for ", num_epochs, " epochs")

        train_losses, val_losses = train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

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
    else:
        # Load the weights from the last epoch
        model.load_state_dict(torch.load("weights/weights_4.pth"))
        print("Weights loaded from the last epoch")



    # Predict on test images
    print("Predicting on test images")
    predict(model, test_image_dir="data/test_images", output_dir="data/test_results", device=device)

    #predict on a training image
    print("Predicting on training images")
    predict(model, test_image_dir="data/train_images", output_dir="data/dump", device=device)



if __name__ == "__main__":
    main()