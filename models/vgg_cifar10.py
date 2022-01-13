'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init

from models.utils_approx import ReLU_maker
from models.utils_approx import maxpool_maker

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, cfg, approx_param_dict_list, batch_norm=False):
        super(VGG, self).__init__()
        relu_dict = approx_param_dict_list[0]
        maxpool_dict = approx_param_dict_list[1]
        maxpool_basic_dict = {'kernel_size': 2, 'stride': 2, 'padding': 0}

        # in_channels = 3

        self.cfg = cfg
        self.batch_norm = batch_norm
        self.relu = ReLU_maker(relu_dict)
        self.maxpool = maxpool_maker(maxpool_dict, maxpool_basic_dict)

        # self.conv2d64 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        # self.conv2d6464 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.conv2d64128 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.conv2d128 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        # self.conv2d256 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        # self.conv2d512 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)

        # self.bn64 = nn.BatchNorm2d(64)
        # self.bn128 = nn.BatchNorm2d(128)
        # self.bn256 = nn.BatchNorm2d(256)
        # self.bn512 = nn.BatchNorm2d(512)

        self.features = self._make_layer(cfg, batch_norm)
        self.drop = nn.Dropout()
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 10)

        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Linear(512, 10),
        # )
          # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def _make_layer(self, cfg, batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [self.maxpool]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), self.relu]
                else:
                    layers += [conv2d, self.relu]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        # x = self.classifier(x)
        return x


def make_layers(cfg, approx_param_dict_list, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11(approx_param_dict_list=[{'type': 'pure'}]*2):
    """VGG 11-layer model (configuration "A")"""
    return VGG(cfg['A'], approx_param_dict_list)


def vgg11_bn(approx_param_dict_list=[{'type': 'pure'}]*2):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(cfg['A'], approx_param_dict_list, batch_norm=True)


def vgg13(approx_param_dict_list=[{'type': 'pure'}]*2):
    """VGG 13-layer model (configuration "B")"""
    return VGG(cfg['B'], approx_param_dict_list)


def vgg13_bn(approx_param_dict_list=[{'type': 'pure'}]*2):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(cfg['B'], approx_param_dict_list, batch_norm=True)


def vgg16(approx_param_dict_list=[{'type': 'pure'}]*2):
    """VGG 16-layer model (configuration "D")"""
    return VGG(cfg['D'], approx_param_dict_list)


def vgg16_bn(approx_param_dict_list=[{'type': 'pure'}]*2):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(cfg['D'], approx_param_dict_list, batch_norm=True)


def vgg19(approx_param_dict_list=[{'type': 'pure'}]*2):
    """VGG 19-layer model (configuration "E")"""
    return VGG(cfg['E'], approx_param_dict_list)


def vgg19_bn(approx_param_dict_list=[{'type': 'pure'}]*2):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(cfg['E'], approx_param_dict_list, batch_norm=True)