import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader

def get_loader(datasets: str, batch_size: int):
    if datasets == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
    elif datasets == 'cifar10':
        # 定义 transform，包括缩放、中心裁剪、随机水平翻转、归一化
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # 只需要归一化和中心裁剪
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # 加载 CIFAR10 数据集
        train_dataset = CIFAR10(root='./data', train=True,download=True, transform=transform_train)
        test_dataset = CIFAR10(root='./data', train=False,download=True, transform=transform_test)
    else:
        raise ValueError(f"datasets should be 'mnist' or 'cifar10', but got {datasets}.")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader