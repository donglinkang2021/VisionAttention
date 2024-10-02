from .resnet import ResNet, BasicBlock, Bottleneck
import torch.nn as nn

from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet34_Weights, 
    ResNet50_Weights, 
    ResNet101_Weights, 
    ResNet152_Weights
)

PRETRAINED_MODELS = {
    'resnet18': {
        'model': ResNet(BasicBlock, [2, 2, 2, 2]),
        'weights': ResNet18_Weights,
    },
    'resnet34': {
        'model': ResNet(BasicBlock, [3, 4, 6, 3]),
        'weights': ResNet34_Weights,
    },
    'resnet50': {
        'model': ResNet(Bottleneck, [3, 4, 6, 3]),
        'weights': ResNet50_Weights,
    },
    'resnet101': {
        'model': ResNet(Bottleneck, [3, 4, 23, 3]),
        'weights': ResNet101_Weights,
    },
    'resnet152': {
        'model': ResNet(Bottleneck, [3, 8, 36, 3]),
        'weights': ResNet152_Weights,
    },
}

def get_pretrained_model(model_name: str):
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Invalid model name: {model_name}")
    model:nn.Module = PRETRAINED_MODELS[model_name]['model']

    from torchvision.models._api import WeightsEnum
    weights:WeightsEnum = PRETRAINED_MODELS[model_name]['weights']
    weights = weights.verify(weights.DEFAULT)
    model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
    
    return model

def get_model(model_name:str, num_classes:int) -> nn.Module:
    if model_name == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif model_name == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    elif model_name == 'resnet50':
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    elif model_name == 'resnet101':
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    elif model_name == 'resnet152':
        return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    else:
        raise ValueError(f"Invalid model name: {model_name}")