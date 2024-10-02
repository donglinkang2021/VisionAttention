from .mnist import MNISTDataModule
from .cifar10 import CIFAR10DataModule
from .cifar100 import CIFAR100DataModule

def get_datamodule(dataset_name: str, data_dir: str, batch_size: int, num_workers: int):
    if dataset_name == 'mnist':
        return MNISTDataModule(data_dir, batch_size, num_workers)
    elif dataset_name == 'cifar10':
        return CIFAR10DataModule(data_dir, batch_size, num_workers)
    elif dataset_name == 'cifar100':
        return CIFAR100DataModule(data_dir, batch_size, num_workers)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
