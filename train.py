# 训练模型
import torch
import torch.nn as nn
from model import CNN, ResHeads
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
n_head = None

## train config
batch_size = 512
learning_rate = 1e-3
num_epochs = 10
eval_interval = 100
save_begin = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## save config
# model_name = 'CNN'
pretrained_backbone = 'resnet152'
classifier_name = 'linear' if n_head is None else f'heads{n_head}'
model_name = f'{pretrained_backbone}_{classifier_name}'
model_ckpts = save_ckpt(model_name)
print(f"the model checkpoints will be saved at {model_ckpts}.")
# ---------------------

torch.manual_seed(2024)
np.random.seed(2024)

# model = CNN(in_channels, n_channels, n_classes)
model = ResHeads(n_classes, n_head, pretrained_backbone)
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
the model checkpoints will be saved at checkpoints/CNN/2024-03-05_19:44:06.
number of parameters: 4.837002 M
---step 979 {'test_loss': 1.362803328037262, 'test_acc': 0.7113}---                                 
loss: 0.14283181726932526: 100%|████████████████████████████████████████████████████████| 980/980 [02:32<00:00,  6.42batch/s]
the model checkpoints will be saved at checkpoints/resnet18_linear/2024-03-05_19:57:06.
number of parameters: 11.694642 M 
---step 979 {'test_loss': 0.851456618309021, 'test_acc': 0.7986}---                                 
loss: 0.11616778373718262: 100%|████████████████████████████████████████████████████████| 980/980 [02:42<00:00,  6.04batch/s]
the model checkpoints will be saved at checkpoints/resnet34_linear/2024-03-05_20:02:30.
number of parameters: 21.802802 M 
---step 979 {'test_loss': 0.8024520188570022, 'test_acc': 0.8245}---                                
loss: 0.09636842459440231: 100%|████████████████████████████████████████████████████████| 980/980 [02:51<00:00,  5.70batch/s]
the model checkpoints will be saved at checkpoints/resnet50_linear/2024-03-05_20:07:36.
number of parameters: 25.577522 M 
---step 979 {'test_loss': 0.5922027975320816, 'test_acc': 0.8501}---                                
loss: 0.06375502794981003: 100%|████████████████████████████████████████████████████████| 980/980 [03:05<00:00,  5.28batch/s]
the model checkpoints will be saved at checkpoints/resnet101_linear/2024-03-05_20:13:30.
number of parameters: 44.569650 M 
---step 979 {'test_loss': 0.6616580709815025, 'test_acc': 0.848}---                                 
loss: 0.03825755417346954: 100%|████████████████████████████████████████████████████████| 980/980 [02:21<00:00,  6.94batch/s]
the model checkpoints will be saved at checkpoints/resnet152_linear/2024-03-05_20:18:56.
number of parameters: 60.213298 M
---step 979 {'test_loss': 0.630085003376007, 'test_acc': 0.8568}---                                 
loss: 0.03981466963887215: 100%|████████████████████████████████████████████████████████| 980/980 [02:47<00:00,  5.87batch/s]
"""