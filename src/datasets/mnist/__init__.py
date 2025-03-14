import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from src.datasets.mnist.model import cnn, mlp

models = {
    'mlp'                   : mlp.MLP,
    'cnn'                   : cnn.CNN,
}

class DataModule(pl.LightningDataModule):
    """Datamodule for the CIFAR10 dataset."""

    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size if args.batch_size else 1
        self.num_classes = 10
        self.input_dim = (3, 32, 32)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))]
        )
        data_path = str(os.path.join(args.data_root, 'CIFAR10'))
        self.cifar10_train = CIFAR10(
            root=data_path, train=True, download=True, transform=transform
        )
        # Note: do not use mnist_train.datasets, it's the original datasets
        # *before* transformations! The transformations are applied in __getitem__
        randrow = torch.randperm(len(self.cifar10_train))[:args.n_examples]
        subset = Subset(self.cifar10_train, randrow)
        self.X, self.y = self._extract_features_targets(subset)
        self.dataset = TensorDataset(self.X, self.y)

    def train_dataloader(self, num_workers=7):
        """Return the train dataloader for PyTorch Lightning.

        Args:
            num_workers (optional): Defaults to 0.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            persistent_workers=True,
        )

    def _extract_features_targets(self, subset):
        X, y = [], []
        for tup in subset:
            X.append(tup[0])
            y.append(tup[1])
        return torch.stack(X), torch.LongTensor(y)