import torch.nn.functional as F
from torch import nn
from src.models import GenericModel
from argparse import Namespace

class MLP(GenericModel):
    """A Multilayer Perceptron models.

    Default is 1 hidden layer with 50 neurons.
    """
    hidden_dim = 50
    num_hidden_layers = 1
    def __init__(self, args: Namespace):
        super().__init__(args)
        # NOTE: nn.ModuleList is not the same as Sequential,
        # the former doesn't have forward implemented
        if self.num_hidden_layers == 0:
            self.layers = nn.Linear(args.input_dim, args.num_classes)
        else:
            self.layers = nn.Sequential(nn.Linear(args.input_dim, self.hidden_dim), nn.ReLU())
            n_layers = 2
            for _ in range(self.num_hidden_layers - 1):
                self.layers.add_module(
                    name=f"{n_layers}", module=nn.Linear(self.hidden_dim, self.hidden_dim)
                )
                self.layers.add_module(name=f"{n_layers+1}", module=nn.ReLU())
                n_layers += 2

            self.layers.add_module(
                name=f"{n_layers}", module=nn.Linear(self.hidden_dim, args.num_classes)
            )

    def forward(self, x_in, apply_softmax=False):
        """Forward pass."""
        # Pytorch lightning recommends using forward for inference, not training
        y_pred = self.layers(x_in)
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred