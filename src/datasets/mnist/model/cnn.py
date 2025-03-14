from src.models import GenericModel
import torch.nn as nn

class CNN(GenericModel):
    def __init__(self, args):
        super().__init__(args)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        self.head = nn.Linear(512, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x