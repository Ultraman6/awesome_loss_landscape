import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from benchmarks.models import vgg
from benchmarks.models import cnn, resnet, densenet
import numpy as np

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

train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

class DataModule:
    """Datamodule for the CIFAR10 dataset."""

    def __init__(self, args):
        super().__init__()
        self.batch_size = getattr(args, 'batch_size', 64)
        self.num_workers = getattr(args, 'num_workers', 0)
        self.train = getattr(args, 'train', True)
        self.num_classes = 100
        self.input_dim = (3, 32, 32)
        data_root = getattr(args, 'data_root', './RAWDATA')
        data_path = str(os.path.join(data_root, 'CIFAR10'))
        if self.train:
            self.dataset = CIFAR100(
                root=data_path, train=True, download=True, transform=train_transform
            )
        else:
            self.dataset = CIFAR100(
                root=data_path, train=False, download=True, transform=test_transform
            )

    def _dataloader(self):
        """Return the train dataloader for PyTorch Lightning.

        Args:
            num_workers (optional): Defaults to 0.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def _data(self, cls_idxes:list=None):
        X, Y = self.dataset.data, self.dataset.targets
        if cls_idxes:
            return X, Y
        else:
            filter = np.where(Y in cls_idxes)
            return X[filter], Y[filter]


