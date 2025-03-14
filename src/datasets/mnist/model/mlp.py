from src.models import GenericModel
import torch.nn as nn

class MLP(GenericModel):
    def __init__(self, args):
        super().__init__(args)
        self.encoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(784, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU()
        )
        self.head = nn.Linear(200, 10)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x