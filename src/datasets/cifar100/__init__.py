import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from src.models import cnn, resnet, vgg, densenet

models = {
    'cnn'                   : cnn.CNN,
    'lenet'                 : cnn.LeNet,
    'vgg9'                  : vgg.VGG9,
    'densenet121'           : densenet.DenseNet121,
    'resnet18'              : resnet.ResNet18,
    'resnet18_noshort'      : resnet.ResNet18_noshort,
    'resnet34'              : resnet.ResNet34,
    'resnet34_noshort'      : resnet.ResNet34_noshort,
    'resnet50'              : resnet.ResNet50,
    'resnet50_noshort'      : resnet.ResNet50_noshort,
    'resnet101'             : resnet.ResNet101,
    'resnet101_noshort'     : resnet.ResNet101_noshort,
    'resnet152'             : resnet.ResNet152,
    'resnet152_noshort'     : resnet.ResNet152_noshort,
    'resnet20'              : resnet.ResNet20,
    'resnet20_noshort'      : resnet.ResNet20_noshort,
    'resnet32_noshort'      : resnet.ResNet32_noshort,
    'resnet44_noshort'      : resnet.ResNet44_noshort,
    'resnet50_16_noshort'   : resnet.ResNet50_16_noshort,
    'resnet56'              : resnet.ResNet56,
    'resnet56_noshort'      : resnet.ResNet56_noshort,
    'resnet110'             : resnet.ResNet110,
    'resnet110_noshort'     : resnet.ResNet110_noshort,
    'wrn56_2'               : resnet.WRN56_2,
    'wrn56_2_noshort'       : resnet.WRN56_2_noshort,
    'wrn56_4'               : resnet.WRN56_4,
    'wrn56_4_noshort'       : resnet.WRN56_4_noshort,
    'wrn56_8'               : resnet.WRN56_8,
    'wrn56_8_noshort'       : resnet.WRN56_8_noshort,
    'wrn110_2_noshort'      : resnet.WRN110_2_noshort,
    'wrn110_4_noshort'      : resnet.WRN110_4_noshort,
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
        data_path = str(os.path.join(args.data_root, 'CIFAR100'))
        self.cifar10_train = CIFAR100(
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