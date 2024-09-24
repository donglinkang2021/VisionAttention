import torch.nn as nn
from typing import Optional, Callable
from torch import Tensor

from .conv2d import conv3x3, conv1x1

class BasicBlock(nn.Module):
    """
    implementation of `BasicBlock` block, refer to `torchvision.models.resnet.BasicBlock`

    - both **self.conv1** and **self.downsample** layers downsample the input when stride != 1
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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


class Bottleneck(nn.Module):
    """
    implementation of `Bottleneck` block, refer to `torchvision.models.resnet.Bottleneck`

    - places the stride for downsampling at 3x3 convolution `self.conv2`
    - both **self.conv2** and **self.downsample** layers downsample the input when stride != 1
    """
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
if __name__ == '__main__':
    import torch
    # Test BasicBlock
    basic_block = BasicBlock(inplanes=64, planes=64)
    print(basic_block)
    x = torch.randn(1, 64, 56, 56)
    y = basic_block(x)
    print(y.size())
    
    # Test Bottleneck
    # just simulate the resnet50
    def downsample(inplanes, planes, stride):
        return nn.Sequential(
            conv1x1(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
        )
    x = torch.randn(1, 64, 56, 56)
    print(x.size())
    layer1 = nn.Sequential(
        Bottleneck(inplanes=64, planes=64, downsample=downsample(64, 256, 1)),
        Bottleneck(inplanes=256, planes=64),
        Bottleneck(inplanes=256, planes=64),
    )
    x = layer1(x)
    print(x.size())
    layer2 = nn.Sequential(
        Bottleneck(inplanes=256, planes=128, stride=2, downsample=downsample(256, 512, 2)),
        Bottleneck(inplanes=512, planes=128),
        Bottleneck(inplanes=512, planes=128),
        Bottleneck(inplanes=512, planes=128),
    )
    x = layer2(x)
    print(x.size())
    layer3 = nn.Sequential(
        Bottleneck(inplanes=512, planes=256, stride=2, downsample=downsample(512, 1024, 2)),
        Bottleneck(inplanes=1024, planes=256),
        Bottleneck(inplanes=1024, planes=256),
        Bottleneck(inplanes=1024, planes=256),
        Bottleneck(inplanes=1024, planes=256),
        Bottleneck(inplanes=1024, planes=256),
    )
    x = layer3(x)
    print(x.size())
    layer4 = nn.Sequential(
        Bottleneck(inplanes=1024, planes=512, stride=2, downsample=downsample(1024, 2048, 2)),
        Bottleneck(inplanes=2048, planes=512),
        Bottleneck(inplanes=2048, planes=512),
    )
    x = layer4(x)
    print(x.size())


# python -m src.models.components.blocks