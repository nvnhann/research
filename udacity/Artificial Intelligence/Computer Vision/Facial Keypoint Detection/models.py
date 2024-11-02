import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 5)  # (1, 224, 224) -> (32, 220, 220)
        self.conv2 = nn.Conv2d(32, 64, 3)  # (32, 110, 110) -> (64, 108, 108)
        self.conv3 = nn.Conv2d(64, 128, 3)  # (64, 54, 54) -> (128, 52, 52)
        self.conv4 = nn.Conv2d(128, 256, 3)  # (128, 26, 26) -> (256, 24, 24)
        self.conv5 = nn.Conv2d(256, 512, 1)  # (256, 12, 12) -> (512, 12, 12)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Reduces dimensions by a factor of 2

        # Fully-connected layers
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 68 * 2)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        # Apply conv layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))  # (32, 220, 220) -> (32, 110, 110)
        x = self.pool(F.relu(self.conv2(x)))  # (64, 108, 108) -> (64, 54, 54)
        x = self.pool(F.relu(self.conv3(x)))  # (128, 52, 52) -> (128, 26, 26)
        x = self.pool(F.relu(self.conv4(x)))  # (256, 24, 24) -> (256, 12, 12)
        x = self.pool(F.relu(self.conv5(x)))  # (512, 12, 12) -> (512, 6, 6)

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Final output layer
        x = self.fc3(x)

        return x
