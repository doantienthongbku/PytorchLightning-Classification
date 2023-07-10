import lightning as lt
from torch.utils.data import DataLoader
import os
import glob
from torchvision import transforms

# import random split function from sklearn
from sklearn.model_selection import train_test_split

from .dataset import CatDogDataset


class LitDataModule(lt.LightningDataModule):
    def __init__(self,
                 data_dir="cats_dogs_light",
                 split_ratio=0.2,
                 batch_size=32,
                 num_worker=4,
                 prefetch_factor=16) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.prefetch_factor = prefetch_factor
        
        self.dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_worker,
            "prefetch_factor": self.prefetch_factor,
            "pin_memory": True,
        }
    
    def setup(self, stage: str) -> None:
        train_val_file_list = glob.glob(os.path.join(self.data_dir, "*.jpg"))
        self.train_file_list, self.valid_file_list = train_test_split(train_val_file_list, test_size=self.split_ratio)
        self.test_file_list = glob.glob(os.path.join(self.data_dir, "*.jpg"))
        
        self.train_ds = CatDogDataset(file_list=self.train_file_list, transform=self.train_transform)
        self.valid_ds = CatDogDataset(file_list=self.valid_file_list, transform=self.valid_transform)
        self.test_ds = CatDogDataset(file_list=self.test_file_list, transform=self.test_transform)
        
        # print information of dataset
        print("=========================================================")
        print(f"Train dataset size: {len(self.train_ds)}")
        print(f"Valid dataset size: {len(self.valid_ds)}")
        print(f"Test dataset size: {len(self.test_ds)}")
        print("=========================================================")
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, **self.dataloader_kwargs)
    
    def val_dataloader(self):
        return DataLoader(self.valid_ds, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, **self.dataloader_kwargs)
        
    def _modify_transform(self):
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.valid_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])