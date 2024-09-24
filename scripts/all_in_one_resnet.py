# save
from pathlib import Path

# model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# training progress
from tqdm import tqdm

# data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

OUT_ROOT = "/root/autodl-tmp/output"
DATA_ROOT = "/root/autodl-tmp/data"
Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_loader(root: str, batch_size: int):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = CIFAR10(root, train=True,download=True, transform=transform_train)
    test_dataset = CIFAR10(root, train=False,download=True, transform=transform_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

@torch.no_grad()
def validate(net:nn.Module, dataloader:DataLoader, loss_fn:nn.Module):
    net.eval()
    metrics = {'loss':0, 'acc':0, 'num_samples':0, 'num_batches':0}
    pbar = tqdm(total=len(dataloader), desc=f"Eval test processing", leave=False)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = net(images)
        loss = loss_fn(output, labels)
        metrics['loss'] += loss.item()
        pred = output.argmax(dim=1)
        metrics['acc'] += (pred==labels).sum().item()
        metrics['num_samples'] += len(labels)
        metrics['num_batches'] += 1
        pbar.update(1)
    pbar.close()
    net.train()
    metrics['loss'] /= metrics['num_batches']
    metrics['acc'] /= metrics['num_samples']
    return metrics

def save_model(net:nn.Module, path:str):
    torch.save(net.state_dict(), path)

def train(net:nn.Module, train_loader:DataLoader, test_loader:DataLoader,
            loss_fn:nn.Module,
            optimizer:torch.optim.Optimizer=None,
            num_epochs:int=5):
    n_batches = len(train_loader)
    best_acc = 0
    max_steps = num_epochs * n_batches
    pbar = tqdm(total = max_steps, desc="Training batches", leave=True, unit="batch")
    interval_eval = max_steps // 10
    for epoch in range(num_epochs):
        for i, (images,labels) in enumerate(train_loader):
            i_batch = epoch * n_batches + i
            if i_batch % interval_eval == 0 or i_batch == max_steps - 1:
                metrics = validate(net, test_loader, loss_fn)
                if i_batch > 0.5 * max_steps and metrics['acc'] > best_acc:
                    best_acc = metrics['acc']
                    save_model(net, f"{OUT_ROOT}/best_model.pth")
            images, labels = images.to(device), labels.to(device)
            output = net(images.to(device))
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(
                loss=loss.item(), eval_loss=metrics['loss'], 
                eval_acc=metrics['acc'], best_acc=best_acc
            )
            pbar.update(1)
    pbar.close()

if __name__ == "__main__":
    # set weight download path
    import os
    from pathlib import Path
    outdir = "/root/autodl-tmp/.cache/torch"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_HOME'] = outdir

    import torchvision.models as models
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet50 = resnet50.to(device)

    n_classes = 10
    in_features = resnet50.fc.in_features
    resnet50.fc = nn.Linear(in_features, n_classes).to(device)

    # for p in resnet50.parameters():
    #     p.requires_grad = False

    # for p in resnet50.fc.parameters():
    #     p.requires_grad = True

    resnet50.load_state_dict(torch.load(f"{OUT_ROOT}/best_model.pth"))
    for p in resnet50.parameters():
        p.requires_grad = True

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(resnet50.parameters(), learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 2
    # batch_size = 512
    batch_size = 128
    trainloader, testloader = get_loader(DATA_ROOT, batch_size)
    train(
        resnet50, trainloader, testloader,
        loss_fn, optimizer, num_epochs
    )

# train fc layer only: batch_size=512 [06:08<00:00,  1.88batch/s, best_acc=0.736, eval_acc=0.735, eval_loss=0.804, loss=0.891] 
# train all layers: batch_size=128 [07:45<00:00,  1.68batch/s, best_acc=0.897, eval_acc=0.886, eval_loss=0.357, loss=0.289]