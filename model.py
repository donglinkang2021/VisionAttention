import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        raise NotImplementedError
    
def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CNN(BaseModel):
    def __init__(self, in_channels: int, n_channels: List[int], n_classes: int):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                n_channels, 
                kernel_size = 7, 
                stride = 2, 
                padding = 3           
            ),
            nn.BatchNorm2d(n_channels), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.res1 = BasicBlock(n_channels, 2 * n_channels, stride=2)
        self.res2 = BasicBlock(2 * n_channels, 4 * n_channels, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * n_channels, n_classes)
        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()/1e6:.6f} M ")

    def forward(self, x):
        x = self.downsample(x)  # B, 3, 32, 32  -> B, 64, 16, 16
        x = self.res1(x)        # B, 64, 16, 16 -> B, 128, 8, 8
        x = self.res2(x)        # B, 128, 8, 8  -> B, 256, 4, 4
        x = self.avgpool(x)     # B, 256, 4, 4  -> B, 256, 1, 1
        x = self.flatten(x)     # B, 256, 1, 1  -> B, 256
        x = self.fc(x)          # B, 256        -> B, 10
        return x