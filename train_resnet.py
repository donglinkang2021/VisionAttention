# 训练模型
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from model import ResHeads
import numpy as np

# config
pretrained_model = 'resnet18'
is_pretrained = False
in_channels = 3
n_channels = 64
n_classes = 10
batch_size = 128
learning_rate = 1e-3
num_epochs = 20
eval_interval = 10
save_begin = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---------------------

torch.manual_seed(2024)
np.random.seed(2024)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
train_dataset = CIFAR10(root='./data', train=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = ResHeads(n_classes, pretrained_model)
model.freeze()
model.to(device)
if is_pretrained:
    model.load_state_dict(torch.load(f'checkpoint/best_{pretrained_model}.pth'))

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate():
    metrics = {}
    model.eval()
    for name, loader in [('train', train_loader), ('test', test_loader)]:
        losses = []
        num_samples = 0
        num_correct = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            losses.append(loss.item())
            num_samples += x.shape[0]
            num_correct += (y_pred.argmax(1) == y).sum().item()
        metrics[name + '_loss'] = np.mean(losses)
        metrics[name + '_acc'] = num_correct / num_samples
    model.train()
    return metrics

# 训练
best_acc = 0
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

            if iter > save_begin and metrics['test_acc'] > best_acc:
                best_acc = metrics['test_acc']
                torch.save(model.state_dict(), f'checkpoint/best_{pretrained_model}.pth')

        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

"""output:
(GPT) root@test:~/VisionAttention# python train_resnet.py 
number of parameters: 11.694642 M 
...
step 130: train_loss: 1.7037 train_acc: 0.4155 test_loss: 1.7273 test_acc: 0.4040 
step 140: train_loss: 1.7110 train_acc: 0.4103 test_loss: 1.7396 test_acc: 0.3930 
step 150: train_loss: 1.6809 train_acc: 0.4223 test_loss: 1.7090 test_acc: 0.4137 
step 160: train_loss: 1.6698 train_acc: 0.4273 test_loss: 1.7008 test_acc: 0.4100 
step 170: train_loss: 1.6863 train_acc: 0.4153 test_loss: 1.7185 test_acc: 0.3981 
step 180: train_loss: 1.6755 train_acc: 0.4220 test_loss: 1.7115 test_acc: 0.4096 
step 190: train_loss: 1.6628 train_acc: 0.4281 test_loss: 1.6969 test_acc: 0.4119 
step 200: train_loss: 1.6610 train_acc: 0.4263 test_loss: 1.6977 test_acc: 0.4078 
"""