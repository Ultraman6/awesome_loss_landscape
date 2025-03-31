from argparse import Namespace
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_noshortcut(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_noshortcut, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_noshortcut(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_noshortcut, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion*planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args:Namespace):
        super().__init__()
        self.in_planes = 64
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            self._make_layer(block, 64, num_blocks[0], stride=1),
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block, 512, num_blocks[3], stride=2),
            nn.AvgPool2d(4),
            nn.Flatten()
        )
        self.head = nn.Sequential(nn.Linear(512*block.expansion, args.num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.encoder(x)
        out = self.head(out)
        return out

class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, args:Namespace):
        super().__init__()
        self.in_planes = 16
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            self._make_layer(block, 16, num_blocks[0], stride=1),
            self._make_layer(block, 32, num_blocks[1], stride=2),
            self._make_layer(block, 64, num_blocks[2], stride=2),
            nn.AvgPool2d(8),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(64 * block.expansion, args.num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.encoder(x)
        out = self.head(out)
        return out

class WResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, k, args:Namespace):
        super().__init__()
        self.in_planes = 16*k
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16*k, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16*k),
            nn.ReLU(inplace=True),
            self._make_layer(block, 16*k, num_blocks[0], stride=1),
            self._make_layer(block, 32*k, num_blocks[1], stride=2),
            self._make_layer(block, 64*k, num_blocks[2], stride=2),
            nn.AvgPool2d(8),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(64*k*block.expansion, args.num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.encoder(x)
        out = self.head(out)
        return out

# ImageNet model
def ResNet18(args):
    return ResNet(BasicBlock, [2, 2, 2, 2], args)

def ResNet18_noshort(args):
    return ResNet(BasicBlock_noshortcut, [2, 2, 2, 2], args)

def ResNet34(args):
    return ResNet(BasicBlock, [3, 4, 6, 3], args)

def ResNet34_noshort(args):
    return ResNet(BasicBlock_noshortcut, [3, 4, 6, 3], args)

def ResNet50(args):
    return ResNet(Bottleneck, [3, 4, 6, 3], args)

def ResNet50_noshort(args):
    return ResNet(Bottleneck_noshortcut, [3, 4, 6, 3], args)

def ResNet101(args):
    return ResNet(Bottleneck, [3, 4, 23, 3], args)

def ResNet101_noshort(args):
    return ResNet(Bottleneck_noshortcut, [3, 4, 23, 3], args)

def ResNet152(args):
    return ResNet(Bottleneck, [3, 8, 36, 3], args)

def ResNet152_noshort(args):
    return ResNet(Bottleneck_noshortcut, [3, 8, 36, 3], args)

# CIFAR-10 model
def ResNet20(args):
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n, n, n], args)

def ResNet20_noshort(args):
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n, n, n], args)

def ResNet32_noshort(args):
    depth = 32
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n, n, n], args)

def ResNet44_noshort(args):
    depth = 44
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n, n, n], args)

def ResNet50_16_noshort(args):
    depth = 50
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n, n, n], args)

def ResNet56(args):
    depth = 56
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n, n, n], args)

def ResNet56_noshort(args):
    depth = 56
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n, n, n], args)

def ResNet110(args):
    depth = 110
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n, n, n], args)

def ResNet110_noshort(args):
    depth = 110
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n, n, n], args)

def WRN56_2(args):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n, n, n], 2, args)

def WRN56_4(args):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n, n, n], 4, args)

def WRN56_8(args):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n, n, n], 8, args)

def WRN56_2_noshort(args):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n, n, n], 2, args)

def WRN56_4_noshort(args):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n, n, n], 4, args)

def WRN56_8_noshort(args):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n, n, n], 8, args)

def WRN110_2_noshort(args):
    depth = 110
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n, n, n], 2, args)

def WRN110_4_noshort(args):
    depth = 110
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n, n, n], 4, args)