import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, List, Optional, Callable, Union
from .components.conv2d import conv3x3, conv1x1
from .components.blocks import BasicBlock, Bottleneck

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int, blocks: int, stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        planes_expaned = planes * block.expansion
        if stride != 1 or self.inplanes != planes_expaned:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes_expaned, stride),
                norm_layer(planes_expaned),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        for _ in range(1, blocks):
            layers.append(block(planes_expaned, planes, norm_layer=norm_layer))
        self.inplanes = planes_expaned
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    # set weight download path
    import os
    from pathlib import Path
    outdir = "/root/autodl-tmp/.cache/torch"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_HOME'] = outdir
    # Test ResNet
    from torchvision.models.resnet import (
        ResNet18_Weights,
        ResNet34_Weights, 
        ResNet50_Weights, 
        ResNet101_Weights, 
        ResNet152_Weights
    )
    from src.utils.show import show_model_size
    data = [
        {
            'name': 'resnet18',
            'model': ResNet(BasicBlock, [2, 2, 2, 2]),
            'weights': ResNet18_Weights,
        },
        {
            'name': 'resnet34',
            'model': ResNet(BasicBlock, [3, 4, 6, 3]),
            'weights': ResNet34_Weights,
        },
        {
            'name': 'resnet50',
            'model': ResNet(Bottleneck, [3, 4, 6, 3]),
            'weights': ResNet50_Weights,
        },
        {
            'name': 'resnet101',
            'model': ResNet(Bottleneck, [3, 4, 23, 3]),
            'weights': ResNet101_Weights,
        },
        {
            'name': 'resnet152',
            'model': ResNet(Bottleneck, [3, 8, 36, 3]),
            'weights': ResNet152_Weights,
        },
    ]
    from torchvision.models._api import WeightsEnum
    for sample in data:
        print(f"Testing {sample['name']}")
        # load pretrained weights
        name = sample['name']
        model:nn.Module = sample['model']
        weights:WeightsEnum = sample['weights']
        weights = weights.verify(weights.DEFAULT)
        model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        print(f"{name}: {y.size()}")
        show_model_size(model)

# python -m src.models.resnet