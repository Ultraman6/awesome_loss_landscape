from argparse import Namespace
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import torchmetrics

def configure_criterion(args: Namespace):
    criterion = getattr(args, 'criterion', 'ce')
    if criterion == "ce":
        return F.cross_entropy
    elif criterion == "bce":
        return F.binary_cross_entropy
    elif criterion == "mse":
        return F.mse_loss
    else:
        raise Exception(f"Criterion not recognized: {criterion}")

def configure_optimizer(model, args: Namespace):
    """Configure the optimizer for Pytorch Lightning.

    Raises:
        Exception: Optimizer not recognized.
    """
    optimizer = getattr(args, 'optimizer', 'sgd')
    lr = getattr(args, 'lr', 0.01)
    weight_decay = getattr(args, 'weight_decay', 0)
    momentum = getattr(args, 'momentum', 0)
    if optimizer == "sgd":
        return SGD(model.parameters(), lr, momentum-momentum, weight_decay=weight_decay)
    elif optimizer == "adam":
        return Adam(model.parameters(), lr, weight_decay=weight_decay)
    else:
        raise Exception(
            f"custom_optimizer supplied is not supported: {optimizer}"
        )

class GenericModel(nn.Module):
    """GenericModel class that enables flattening of the models parameters."""

    def __init__(self, args: Namespace):
        super().__init__()
        self.optimizer = configure_optimizer(self, args)
        self.criterion = configure_criterion(args)
        self.device = args.device
        self.num_classes = args.num_classes
        self.optim_path = []
        self.acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=args.num_classes
        )

    def _step(self, batch):
        x, y = batch
        preds = self(x)
        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()
        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        return loss, acc