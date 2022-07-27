from msilib.schema import Class
from typing import Callable, Optional, List
import torch
from torch import Tensor, nn

from typing import Optional, Callable

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding from torchvision"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution from torchvision"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResidualBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, base_width = 64, groups: int = 1, dilation: int = 1, 
    downsample:Optional[nn.Module] = None, norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(ResidualBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("ResidualBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 is not supported by ResidualBlock")

        self.relu = nn.ReLU()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)      

        self.downsample = downsample  

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(self, block: ResidualBlock, layers: List[int],  zero_init_residual: bool = False, groups: int = 1, 
                    width_per_group: int = 64, replace_stride_with_dilation: Optional[List[bool]] = None,
                    norm_layer:Optional[Callable[..., nn.Module]] = None):
        
        super(ResNet, self).__init__()

        self.norm_layer = norm_layer
        if self.norm_layer is None:
            self.norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None for a 3-element tuple, got {replace_stride_with_dilation}")

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if zero_init_residual:
                if isinstance(m, ResidualBlock) and m.bn2.weigth is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: ResidualBlock, planes: int, blocks: int, stride: int = 1, dilate: bool = False ) -> nn.Sequential:
        norm_layer = self.norm_layer
        downsample = None
        prev_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )
            
        layers = list()
        layers.append(block(self.inplanes, planes, stride, self.base_width, self.groups, prev_dilation, downsample, norm_layer))
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor :
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class Classifier(nn.Module):
    def __init__(self, block: ResidualBlock, num_classes: int = 1000):
        super(Classifier, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(block: ResidualBlock, zero_init_residual: bool = False, groups: int = 1, width_per_group: int = 64,
                replace_stride_with_dilation: Optional[List[bool]] = None, 
                norm_layer: Optional[Callable[..., nn.Module]] = None) -> ResNet:
    return ResNet(block, [2, 2, 2, 2], zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)

import os

current = os.path.dirname(os.path.realpath(__file__))  
BACKBONE_WEIGTHS_PATH = current + r"\resnet18.pth"
CLASSIFIER_WEIGTHS_PATH = current + r"\resnet18_classifier.pth"
LOAD_WEIGTHS = True
SAVE_CUR_WEIGTHS = True

if __name__ == "__main__":    
    block = ResidualBlock

    backbone = resnet18(block)
    classifier = Classifier(block)
    
    if LOAD_WEIGTHS:
        if(os.path.exists(BACKBONE_WEIGTHS_PATH)):
            backbone.load_state_dict(torch.load(BACKBONE_WEIGTHS_PATH))
        if(os.path.exists(CLASSIFIER_WEIGTHS_PATH)):
            classifier.load_state_dict(torch.load(CLASSIFIER_WEIGTHS_PATH))

    

