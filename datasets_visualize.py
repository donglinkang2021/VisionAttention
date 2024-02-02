from torchvision.datasets import MNIST, CIFAR10
import matplotlib.pyplot as plt


# 使用MNIST类加载数据集

train_dataset = MNIST(root='./data', train=True, download=False)
test_dataset = MNIST(root='./data', train=False, download=False)

from draw import show_attention_batch
show_attention_batch(
    train_dataset.data[:9], 
    xlabel="rows",
    ylabel="cols",
    title="Intensity",
    figsize=(8, 8), 
    cmap='gray'
)


# 使用CIFAR10类加载数据集

train_dataset = CIFAR10(root='./data', train=True, download=False)
test_dataset = CIFAR10(root='./data', train=False, download=False)

# 显示CIFAR10数据集的图像
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(train_dataset.data[i])
    plt.title(train_dataset.classes[train_dataset.targets[i]])
    plt.axis('off')
plt.show()