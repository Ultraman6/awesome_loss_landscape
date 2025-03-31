import argparse
import os, sys
root = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(root)
from benchmark.main import print_distribution
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from benchmark.model import vgg
from benchmark.model import cnn, resnet, densenet
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

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    def __init__(self, root, train=True, imbalance_ratio=0.01, imb_type='exp'):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform=None, target_transform=None, download=True)
        self.train = train
        if self.train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
            self.gen_imbalanced_data(img_num_list)
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        self.labels = self.targets

        print("{} Mode: Contain {} images".format('train' if train else 'eval', len(self.data)))

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class DataModule:
    """Datamodule for the CIFAR10 dataset."""

    def __init__(self, args):
        super().__init__()
        self.batch_size = getattr(args, 'batch_size', 64)
        self.num_workers = getattr(args, 'num_workers', 0)
        self.cifar_imb_ratio = getattr(args, 'cifar_imb_ratio', 0.01)
        self.num_classes = 10
        self.input_dim = (3, 32, 32)
        data_root = getattr(args, 'data_root', '../../../data')
        self.data_path = str(os.path.join(data_root, 'CIFAR10'))
        self.train_set = IMBALANCECIFAR10(root=self.data_path, imbalance_ratio=self.cifar_imb_ratio, train=True)
        # 平衡评估集
        self.eval_set = IMBALANCECIFAR10(root=self.data_path, imbalance_ratio=1.0, train=False)

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
                shuffle=True
            )
        return self._train_loader

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

    @property
    def eval_loader(self):
        """Return the train dataloader for PyTorch Lightning.
        Args:
            num_workers (optional): Defaults to 0.
        """
        if not hasattr(self, '_eval_loader'):
            self._eval_loader = DataLoader(
                self.eval_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True,
                shuffle=True
            )
        return self._eval_loader

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    dataset = DataModule(args)
    print_distribution(dataset)