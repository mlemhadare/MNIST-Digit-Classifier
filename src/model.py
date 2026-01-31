"""
model.py
CNN model definition for MNIST classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    """
    def __init__(self, dropout_rate: float = 0.5) -> None:
        """
        Args:
            dropout_rate (float): Dropout probability for regularization.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (N, 1, 28, 28).
        Returns:
            torch.Tensor: Logits for each class.
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
