from torch import nn
from src.models import GenericModel
from argparse import Namespace

class CNN(GenericModel):
    """CNN convolutional neural network."""

    def __init__(self, args: Namespace):
        super().__init__(args)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(1600, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
        )
        self.head = nn.Linear(192, self.num_classes)

    def forward(self, x):
        """Forward pass."""
        x = self.encoder(x)
        return self.head(x)

class LeNet(GenericModel):
    """LeNet-5 convolutional neural network."""

    def __init__(self, args: Namespace):
        super().__init__(args)

        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        """Forward pass."""
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))  # (n_examples, 120, 1, 1) -> (n_examples, 120)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x