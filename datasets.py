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
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_dataset = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
        test_dataset = CIFAR10(root='./data', train=False, transform=transform_test, download=True)
    else:
        raise ValueError(f"datasets should be 'mnist' or 'cifar10', but got {datasets}.")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader