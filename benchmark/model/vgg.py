from argparse import Namespace
import torch.nn as nn

cfg = {
    'VGG9':  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, args: Namespace):
        super(VGG, self).__init__()
        self.input_size = 32
        self.n_maps = cfg[vgg_name][-2]
        self.encoder = nn.Sequential(
            self._make_layers(cfg[vgg_name]),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            self._make_fc_layers(),
            nn.Linear(self.n_maps, args.num_classes)
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.head(out)
        return out

    def _make_fc_layers(self):
        layers = []
        layers += [nn.Linear(self.n_maps*self.input_size*self.input_size, self.n_maps),
                   nn.BatchNorm1d(self.n_maps),
                   nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
                self.input_size = self.input_size // 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

def VGG9(args):
    return VGG('VGG9', args)

def VGG16(args):
    return VGG('VGG16', args)

def VGG19(args):
    return VGG('VGG19', args)
