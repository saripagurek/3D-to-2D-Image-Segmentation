import torch.nn as nn

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
