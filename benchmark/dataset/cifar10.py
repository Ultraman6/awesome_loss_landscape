import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from benchmark.model import vgg
from benchmark.model import cnn, resnet, densenet

models = {
    'cnn'                   : cnn.CNN,
    'fl_cnn'                : cnn.FLCNN,
    'lenet'                 : cnn.LeNet,
    'vgg9'                  : vgg.VGG9,
    'densenet121'           : densenet.DenseNet121,
    'densenet_cifar'        : densenet.densenet_cifar,
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
# 暂时不要动
train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))
    ]
)

class DataModule:
    """Datamodule for the CIFAR10 dataset."""

    def __init__(self, args):
        super().__init__()
        self.batch_size = getattr(args, 'batch_size', 64)
        self.num_workers = getattr(args, 'num_workers', 1)
        self.num_classes = 10
        self.input_dim = (3, 28, 28)
        self.dim = 1600
        data_root = getattr(args, 'data_root', '../../../data')
        self.data_path = str(os.path.join(data_root, 'CIFAR10'))
        self.train_set = CIFAR10(root=self.data_path, train=True, download=True, transform=train_transform)
        self.eval_set = CIFAR10(root=self.data_path, train=False, download=True, transform=test_transform)

    def train_loader(self, cls_list=None):
        """Return the train dataloader for PyTorch Lightning.

        Args:
            num_workers (optional): Defaults to 0.
        """
        if cls_list:
            return DataLoader(
                TensorDataset(*self.eval_data(cls_list)),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
                shuffle=True
            )
        if not hasattr(self, '_train_loader'):
            self._train_loader = DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
                shuffle=True
            )
        return self._train_loader

    def eval_loader(self, cls_list=None):
        """Return the train dataloader for PyTorch Lightning.
        Args:
            num_workers (optional): Defaults to 0.
        """
        if cls_list:
            return DataLoader(
                TensorDataset(*self.eval_data(cls_list)),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
                shuffle=True
            )
        if not hasattr(self, '_eval_loader'):
            self._eval_loader = DataLoader(
                self.eval_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
                shuffle=True
            )
        return self._eval_loader

    @property
    def train_vol(self):
        if not hasattr(self, '_train_vol'):
            self._train_vol = len(self.train_set)
        return self._train_vol

    @property
    def eval_vol(self):
        if not hasattr(self, '_eval_vol'):
            self._eval_vol = len(self.eval_set)
        return self._eval_vol

    def train_data(self, cls_idxes: list=None):
        X, Y = [], []
        for idx in range(self.train_vol):
            x, y = self.train_set[idx]
            if cls_idxes and y not in cls_idxes:
                continue
            X.append(x)
            Y.append(torch.tensor(y))
        X = torch.stack(X, dim=0)
        Y = torch.stack(Y, dim=0)
        return X, Y

    def eval_data(self, cls_idxes: list=None):
        X, Y = [], []
        for idx in range(self.eval_vol):
            x, y = self.eval_set[idx]
            if cls_idxes and y not in cls_idxes:
                continue
            X.append(x)
            Y.append(torch.tensor(y))
        X = torch.stack(X, dim=0)
        Y = torch.stack(Y, dim=0)
        return X, Y