# 训练模型
import torch
import torch.nn as nn
from models import ResNet, get_num_params
from datasets import get_loader
import numpy as np
import config

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(2024)
np.random.seed(2024)

import torchvision.models as models
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = model.to(device)
n_classes = 10
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, n_classes).to(device)
print(f"number of parameters: {get_num_params(model)/1e6:.6f} M ")
model.to(device)
criterion = nn.CrossEntropyLoss()

pretrained_path = f'dev/best_model.pth'
model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')))


_, test_loader = get_loader('cifar10', config.batch_size)

@torch.no_grad()
def estimate():
    metrics = {}
    model.eval()
    losses = []
    num_samples = 0
    num_correct = 0
    pbar = tqdm(total=len(test_loader), ncols=100, desc=f"Eval test processing", leave=False)
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        losses.append(loss.item())
        num_samples += x.shape[0]
        num_correct += (y_pred.argmax(1) == y).sum().item()
        pbar.update(1)
    metrics['test_loss'] = np.mean(losses)
    metrics['test_acc'] = num_correct / num_samples
    model.train()
    return metrics

metrics = estimate()

print(metrics)