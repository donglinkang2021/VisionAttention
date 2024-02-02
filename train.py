# 训练模型
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from model import CNN
import numpy as np

# config
in_channels=3
n_classes=10
batch_size = 64
learning_rate = 1e-3
num_epochs = 20
eval_interval = 10
# ---------------------

# 定义数据转换
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# # 使用MNIST类加载数据集
# train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
# test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


model = CNN(in_channels, n_classes)

criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate():
    metrics = {}
    model.eval()
    for name, loader in [('train', train_loader), ('test', test_loader)]:
        losses = []
        num_samples = 0
        num_correct = 0
        for x, y in loader:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            losses.append(loss.item())
            num_samples += x.shape[0]
            num_correct += (y_pred.argmax(1) == y).sum().item()
        metrics[name] = np.mean(losses)
        metrics[name + '_acc'] = num_correct / num_samples
    model.train()
    return metrics

# 训练
n_batches = len(train_loader)
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):
        iter = epoch * n_batches + i

        if iter % eval_interval == 0 or iter == num_epochs * n_batches - 1:
            metrics = estimate()
            print(f"step {iter}:", end=' ')
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}", end=' ')
            print()

        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()