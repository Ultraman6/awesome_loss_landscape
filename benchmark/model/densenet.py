import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
    def __init__(self, args:Namespace, block, nblocks, growth_rate=12, reduction=0.5):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes1 = 2*growth_rate
        num_planes2 = num_planes1 + nblocks[0]*growth_rate

        num_planes3 = int(math.floor(num_planes2*reduction))
        num_planes4 = num_planes3 + nblocks[1]*growth_rate

        num_planes5 = int(math.floor(num_planes4*reduction))
        num_planes6 = num_planes5 + nblocks[2]*growth_rate

        num_planes7 = int(math.floor(num_planes6*reduction))
        num_planes8 = num_planes7 + nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes8)
        self.encoder  = nn.Sequential(
            nn.Conv2d(3, num_planes1, kernel_size=3, padding=1, bias=False),
            self._make_dense_layers(block, num_planes1, nblocks[0]),
            Transition(num_planes2, num_planes3),
            self._make_dense_layers(block, num_planes3, nblocks[1]),
            Transition(num_planes4, num_planes5),
            self._make_dense_layers(block, num_planes5, nblocks[2]),
            Transition(num_planes6, num_planes7),
            self._make_dense_layers(block, num_planes7, nblocks[3]),
            nn.BatchNorm2d(num_planes8),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(num_planes8, args.num_classes)
        )

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.encoder(x)
        out = self.head(out)
        return out

def DenseNet121(args):
    return DenseNet(block=Bottleneck, nblocks=[6,12,24,16], growth_rate=32, args=args)

def DenseNet169(args):
    return DenseNet(block=Bottleneck, nblocks=[6,12,32,32], growth_rate=32, args=args)

def DenseNet201(args):
    return DenseNet(block=Bottleneck, nblocks=[6,12,48,32], growth_rate=32, args=args)

def DenseNet161(args):
    return DenseNet(block=Bottleneck, nblocks=[6,12,36,24], growth_rate=48, args=args)

def densenet_cifar(args):
    return DenseNet(block=Bottleneck, nblocks=[6,12,24,16], growth_rate=12, args=args)