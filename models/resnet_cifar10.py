import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from models.utils_approx import ReLU_maker
from models.utils_approx import maxpool_maker


# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#            'wide_resnet50_2', 'wide_resnet101_2']

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def tensor_save_txt(x, name):
    f = open('tempdata/' + name + '.txt', 'w')
    x = x.view(x.size(0), -1)
    for single in list(x):
        for num in list(single):
            f.write(str(num.item()))
    f.close()



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, approx_param_dict_list, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = ReLU_maker(approx_param_dict_list[0])
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, approx_param_dict_list, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.approx_param_dict_list = approx_param_dict_list
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        relu_dict = approx_param_dict_list[0]
        self.relu = ReLU_maker(relu_dict)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.approx_param_dict_list, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(approx_param_dict_list=[{'type': 'pure'}]):
    return ResNet(BasicBlock, [3, 3, 3], approx_param_dict_list)


def resnet32(approx_param_dict_list=[{'type': 'pure'}]):
    return ResNet(BasicBlock, [5, 5, 5], approx_param_dict_list)


def resnet44(approx_param_dict_list=[{'type': 'pure'}]):
    return ResNet(BasicBlock, [7, 7, 7], approx_param_dict_list)


def resnet56(approx_param_dict_list=[{'type': 'pure'}]):
    return ResNet(BasicBlock, [9, 9, 9], approx_param_dict_list)


def resnet110(approx_param_dict_list=[{'type': 'pure'}]):
    return ResNet(BasicBlock, [18, 18, 18], approx_param_dict_list)


def resnet1202(approx_param_dict_list=[{'type': 'pure'}]):
    return ResNet(BasicBlock, [200, 200, 200], approx_param_dict_list)