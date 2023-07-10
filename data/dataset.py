import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import glob
from typing import List

class CatDogDataset(Dataset):
    def __init__(self, 
                 file_list:List=['img1.jpg','img2.jpg'], 
                 transform=None):
        self.file_list = file_list
        self.transform = transform
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        label = 1 if "cat" in os.path.basename(img_path) else 0
        
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            image = self.transform(image)
            
        return image, label