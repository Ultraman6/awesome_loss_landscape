import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from benchmarks.models import vgg
from benchmarks.models import cnn, resnet, densenet
import numpy as np
from datasets import load_dataset

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

class Cifar10_LT(Dataset):
    def __init__(self, root, transform, fix='r-20', train=True):
        self.dataset = load_dataset("tomas-gajarsky/cifar10-lt", name=fix, cache_dir=root)['train' if train else 'test']
        self.transform = transform
        self.data, self.targets = np.array(self.dataset['img']), np.array(self.dataset['label'])
        print(f"image shape is {self.data[0].shape}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.transform(self.data[idx])
        label = self.targets[idx]
        return image, label

class DataModule:
    """Datamodule for the CIFAR10 dataset."""

    def __init__(self, args):
        super().__init__()
        self.batch_size = getattr(args, 'batch_size', 64)
        self.num_workers = getattr(args, 'num_workers', 0)
        self.eval = getattr(args, 'eval', False)
        self.fix = getattr(args, 'data_fix', 'r-20')
        self.num_classes = 10
        self.input_dim = (3, 32, 32)
        data_root = getattr(args, 'data_root', './RAWDATA')
        data_path = str(os.path.join(data_root, 'CIFAR10-LT', self.fix))
        self.train_set = Cifar10_LT(root = data_path, train=True, transform=train_transform)
        if self.eval:
            self.eval_set = Cifar10_LT(root=data_path, train=False, transform=test_transform)

    @property
    def train_loader(self):
        """Return the train dataloader for PyTorch Lightning.

        Args:
            num_workers (optional): Defaults to 0.
        """
        if not hasattr(self, '_train_loader'):
            self._train_loader = DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
            )
        return self._train_loader

    @property
    def train_vol(self):
        if not hasattr(self, '_train_vol'):
            self._train_vol = len(self.train_loader)
        return self._train_vol

    @property
    def eval_vol(self):
        if not hasattr(self, '_eval_vol'):
            self._eval_vol = len(self.eval_loader)
        return self._eval_vol

    @property
    def eval_loader(self):
        """Return the train dataloader for PyTorch Lightning.
        Args:
            num_workers (optional): Defaults to 0.
        """
        assert self.eval
        if not hasattr(self, '_eval_loader'):
            self._eval_loader = DataLoader(
                self.eval_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
            )
        return self._eval_loader

    def _train_data(self, cls_idxes: list=None):
        X, Y = self.train_set.data, self.train_set.targets
        if cls_idxes:
            return X, Y
        else:
            filter = np.where(Y in cls_idxes)
            return X[filter], Y[filter]

    def _eval_data(self, cls_idxes: list=None):
        assert self.eval
        X, Y = self.eval_set.data, self.eval_set.targets
        if cls_idxes:
            return X, Y
        else:
            filter = np.where(Y in cls_idxes)
            return X[filter], Y[filter]

