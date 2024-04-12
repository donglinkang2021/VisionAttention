# 训练模型
import torch
import torch.nn as nn
from models import ResNet, get_num_params, save_ckpt
from datasets import get_loader
import numpy as np
import config

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(2024)
np.random.seed(2024)

model = ResNet(config.n_classes, config.pretrained_backbone, config.is_freeze)
print(f"number of parameters: {get_num_params(model)/1e6:.6f} M ")
model_ckpts = save_ckpt(config.model_name)
print(f"the model checkpoints will be saved at {model_ckpts}.")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
train_loader, test_loader = get_loader('cifar10', config.batch_size)

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

best_acc = 0
n_batches = len(train_loader)
pbar = tqdm(total=config.num_epochs * n_batches, desc="Processing batches", leave=True, unit="batch")
for epoch in range(config.num_epochs):
    for i, (x, y) in enumerate(train_loader):

        iter = epoch * n_batches + i
        if iter % config.eval_interval == 0 or iter == config.num_epochs * n_batches - 1:
            metrics = estimate()
            print(f"\n---step {iter} {metrics}---")

            if iter > config.save_begin and metrics['test_acc'] > best_acc:
                best_acc = metrics['test_acc']
                torch.save(model.state_dict(), f'{model_ckpts}/best_{config.model_name}.pth')

        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss: {loss.item()}")
        pbar.update(1)
pbar.close()