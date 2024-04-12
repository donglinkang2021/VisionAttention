from torchvision.datasets import MNIST, CIFAR10


# 使用MNIST类加载数据集
train_dataset = MNIST(root='./data', train=True, download=True)
test_dataset = MNIST(root='./data', train=False, download=True)
train_dataset = CIFAR10(root='./data', train=True, download=True)
test_dataset = CIFAR10(root='./data', train=False, download=True)