# 训练模型
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from model import ResHeads
import numpy as np

# config
pretrained_backbone = 'resnet18'
is_pretrained = False
n_classes = 10
n_head = None
batch_size = 128
learning_rate = 1e-3
num_epochs = 20
eval_interval = 10
save_begin = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classifier_name = 'linear' if n_head is None else f'heads{n_head}'
model_name = f'{pretrained_backbone}_{classifier_name}'
# ---------------------

torch.manual_seed(2024)
np.random.seed(2024)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
train_dataset = CIFAR10(root='./data', train=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = ResHeads(n_classes, n_head, pretrained_backbone)
# model.freeze()
model.to(device)
if is_pretrained:
    model.load_state_dict(torch.load(f'checkpoint/best_{model_name}.pth'))

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
                torch.save(model.state_dict(), f'checkpoint/best_{model_name}.pth')

        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

"""output:
(GPT) root@test:~/VisionAttention# python train_resnet.py 
number of parameters: 11.694642 M 
step 0: train_loss: 2.4048 train_acc: 0.0780 test_loss: 2.4043 test_acc: 0.0775 
step 10: train_loss: 2.4645 train_acc: 0.2864 test_loss: 2.4402 test_acc: 0.2918 
step 20: train_loss: 2.4188 train_acc: 0.3992 test_loss: 2.3897 test_acc: 0.4030 
step 30: train_loss: 1.3943 train_acc: 0.5425 test_loss: 1.4112 test_acc: 0.5410 
step 40: train_loss: 1.2707 train_acc: 0.5692 test_loss: 1.3154 test_acc: 0.5605 
step 50: train_loss: 1.1562 train_acc: 0.5997 test_loss: 1.1865 test_acc: 0.5912 
step 60: train_loss: 1.0385 train_acc: 0.6488 test_loss: 1.0630 test_acc: 0.6392 
step 70: train_loss: 1.0703 train_acc: 0.6463 test_loss: 1.1088 test_acc: 0.6321 
step 80: train_loss: 0.9399 train_acc: 0.6781 test_loss: 0.9937 test_acc: 0.6634 
step 90: train_loss: 1.0673 train_acc: 0.6483 test_loss: 1.1256 test_acc: 0.6296 
step 100: train_loss: 0.9627 train_acc: 0.6804 test_loss: 1.0155 test_acc: 0.6659 
step 110: train_loss: 0.8716 train_acc: 0.7051 test_loss: 0.9296 test_acc: 0.6925 
step 120: train_loss: 0.9050 train_acc: 0.6896 test_loss: 0.9512 test_acc: 0.6791 
step 130: train_loss: 0.8342 train_acc: 0.7179 test_loss: 0.8933 test_acc: 0.6963 
step 140: train_loss: 0.7777 train_acc: 0.7409 test_loss: 0.8517 test_acc: 0.7249 
step 150: train_loss: 0.7862 train_acc: 0.7377 test_loss: 0.8529 test_acc: 0.7178 
step 160: train_loss: 0.7785 train_acc: 0.7343 test_loss: 0.8603 test_acc: 0.7135 
step 170: train_loss: 0.8263 train_acc: 0.7226 test_loss: 0.9201 test_acc: 0.6943 
step 180: train_loss: 0.7970 train_acc: 0.7318 test_loss: 0.8684 test_acc: 0.7037 
step 190: train_loss: 0.7649 train_acc: 0.7388 test_loss: 0.8410 test_acc: 0.7175 
step 200: train_loss: 0.7502 train_acc: 0.7496 test_loss: 0.8347 test_acc: 0.7225 
step 210: train_loss: 0.7614 train_acc: 0.7356 test_loss: 0.8584 test_acc: 0.7033 
step 220: train_loss: 0.7881 train_acc: 0.7423 test_loss: 0.8648 test_acc: 0.7177 
step 230: train_loss: 0.7972 train_acc: 0.7366 test_loss: 0.8858 test_acc: 0.7136 
step 240: train_loss: 0.7555 train_acc: 0.7418 test_loss: 0.8554 test_acc: 0.7121 
step 250: train_loss: 0.6751 train_acc: 0.7746 test_loss: 0.7735 test_acc: 0.7398 
step 260: train_loss: 0.7413 train_acc: 0.7518 test_loss: 0.8275 test_acc: 0.7255 
step 270: train_loss: 0.6902 train_acc: 0.7693 test_loss: 0.7950 test_acc: 0.7397 
step 280: train_loss: 0.6727 train_acc: 0.7730 test_loss: 0.7719 test_acc: 0.7450 
step 290: train_loss: 0.6790 train_acc: 0.7773 test_loss: 0.7728 test_acc: 0.7476 
step 300: train_loss: 0.6689 train_acc: 0.7719 test_loss: 0.7777 test_acc: 0.7339 
step 310: train_loss: 0.6267 train_acc: 0.7872 test_loss: 0.7395 test_acc: 0.7501 
step 320: train_loss: 0.6615 train_acc: 0.7802 test_loss: 0.7798 test_acc: 0.7434 
step 330: train_loss: 0.6122 train_acc: 0.7975 test_loss: 0.7151 test_acc: 0.7577 
step 340: train_loss: 0.6293 train_acc: 0.7869 test_loss: 0.7338 test_acc: 0.7551 
step 350: train_loss: 0.6934 train_acc: 0.7653 test_loss: 0.8045 test_acc: 0.7295 
step 360: train_loss: 0.6325 train_acc: 0.7857 test_loss: 0.7465 test_acc: 0.7518 
step 370: train_loss: 0.6060 train_acc: 0.7956 test_loss: 0.7303 test_acc: 0.7561 
step 380: train_loss: 0.7157 train_acc: 0.7627 test_loss: 0.8302 test_acc: 0.7321 
step 390: train_loss: 0.6011 train_acc: 0.7974 test_loss: 0.7274 test_acc: 0.7576 
step 400: train_loss: 0.5953 train_acc: 0.8002 test_loss: 0.7235 test_acc: 0.7597 
step 410: train_loss: 0.6133 train_acc: 0.7952 test_loss: 0.7564 test_acc: 0.7523 
step 420: train_loss: 0.6571 train_acc: 0.7784 test_loss: 0.8127 test_acc: 0.7368 
step 430: train_loss: 0.6834 train_acc: 0.7734 test_loss: 0.8325 test_acc: 0.7340 
step 440: train_loss: 0.5756 train_acc: 0.8103 test_loss: 0.7296 test_acc: 0.7642 
step 450: train_loss: 0.5520 train_acc: 0.8088 test_loss: 0.6983 test_acc: 0.7666 
step 460: train_loss: 0.6323 train_acc: 0.7871 test_loss: 0.8000 test_acc: 0.7367 
step 470: train_loss: 0.5420 train_acc: 0.8206 test_loss: 0.7154 test_acc: 0.7632 
step 480: train_loss: 0.5258 train_acc: 0.8241 test_loss: 0.6907 test_acc: 0.7668 
step 490: train_loss: 0.5615 train_acc: 0.8123 test_loss: 0.7213 test_acc: 0.7628 
step 500: train_loss: 0.5405 train_acc: 0.8161 test_loss: 0.7024 test_acc: 0.7678 
step 510: train_loss: 0.6650 train_acc: 0.7758 test_loss: 0.8288 test_acc: 0.7293  
"""