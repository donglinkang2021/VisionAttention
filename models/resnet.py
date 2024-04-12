import torch.nn as nn
import torchvision.models as models

def get_backbone(pretrained_model: str='resnet18'):
    if pretrained_model == 'resnet18':
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif pretrained_model == 'resnet34':
        return models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif pretrained_model == 'resnet50':
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif pretrained_model == 'resnet101':
        return models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif pretrained_model == 'resnet152':
        return models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model: {pretrained_model}")
    
class ResNet(nn.Module):
    def __init__(self, n_classes: int, pretrained_model: str='resnet18', is_freeze: bool=True):
        super().__init__()
        backbone = get_backbone(pretrained_model)
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        if is_freeze:
            self._freeze_feature()
            
        features_dim = backbone.fc.in_features
        self.classifier = nn.Linear(features_dim, n_classes)
        
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)

    def _freeze_feature(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x