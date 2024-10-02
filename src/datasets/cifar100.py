import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import lightning as L

class CIFAR100DataModule(L.LightningDataModule):
    def __init__(self, data_dir:str, batch_size:int, num_workers:int, split_seed:int=1337):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_seed = split_seed
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def prepare_data(self):
        # single gpu
        CIFAR100(root=self.data_dir, train=True, download=True)
        CIFAR100(root=self.data_dir, train=False, download=True)
        
    
    def setup(self, stage=None):
        # multi-gpu
        entire_dataset = CIFAR100(root=self.data_dir, train=True, transform=self.transform_train, download=False)
        self.test_set = CIFAR100(root=self.data_dir, train=False, transform=self.transform_test, download=False)
        train_set_size = int(len(entire_dataset) * 0.8)
        valid_set_size = len(entire_dataset) - train_set_size
        seed = torch.Generator().manual_seed(self.split_seed)
        self.train_set, self.valid_set = data.random_split(entire_dataset, [train_set_size, valid_set_size], generator=seed)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_set, 
            self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False
        )
    