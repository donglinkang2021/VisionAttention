# 训练模型
import torch
import torch.nn as nn
from model import CNN
from datasets import get_loader
import numpy as np
from utils import save_ckpt
from tqdm import tqdm

# config
## model config
is_pretrained = False
in_channels = 3
n_channels = 64
n_classes = 10

## train config
batch_size = 512
learning_rate = 1e-3
num_epochs = 10
eval_interval = 20
save_begin = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## save config
model_name = 'CNN'
model_ckpts = save_ckpt(model_name)
print(f"the model checkpoints will be saved at {model_ckpts}.")
# ---------------------

torch.manual_seed(2024)
np.random.seed(2024)

model = CNN(in_channels, n_channels, n_classes)
model.to(device)
if is_pretrained:
    pretrained_path = f'checkpoint/best_{model_name}.pth'
    model.load_state_dict(torch.load(pretrained_path))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_loader, test_loader = get_loader('cifar10', batch_size)

@torch.no_grad()
def estimate():
    metrics = {}
    model.eval()
    for split, loader in [('test', test_loader)]:
        losses = []
        num_samples = 0
        num_correct = 0
        for x, y in tqdm(loader, ncols=100, desc=f"Eval {split} processing", leave=False):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            losses.append(loss.item())
            num_samples += x.shape[0]
            num_correct += (y_pred.argmax(1) == y).sum().item()
        metrics[split + '_loss'] = np.mean(losses)
        metrics[split + '_acc'] = num_correct / num_samples
    model.train()
    return metrics

best_acc = 0
n_batches = len(train_loader)
pbar = tqdm(total=num_epochs * n_batches, desc="Processing batches", leave=True, unit="batch")
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):

        iter = epoch * n_batches + i
        if iter % eval_interval == 0 or iter == num_epochs * n_batches - 1:
            metrics = estimate()
            print(f"\n---step {iter} {metrics}---")

            if iter > save_begin and metrics['test_acc'] > best_acc:
                best_acc = metrics['test_acc']
                torch.save(model.state_dict(), f'{model_ckpts}/best_{model_name}.pth')

        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss: {loss.item()}")
        pbar.update(1)
pbar.close()

"""output:
---step 979 {'test_loss': 1.362803328037262, 'test_acc': 0.7113}---                                 
loss: 0.14283181726932526: 100%|████████████████████████████████████████████████████████| 980/980 [02:32<00:00,  6.42batch/s]
"""