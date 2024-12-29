import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import os
import numpy as np

class PixelClassDataset(torch.utils.data.Dataset):
    """
    custom Dataset for CIFAR10 for use in classification colorizing problem
    """
    
    def __init__(self, root, train=True, transform=None):
        self.data = torchvision.datasets.CIFAR10(root=root, train=train, download=False)
        self.transform = transform

    def __getitem__(self, index):
        img, _ = self.data[index]
        img_tensor = transforms.ToTensor()(img)  # Convert to tensor
        label_tensor = (img_tensor * 255).long()  # Create label tensor
        return img_tensor, label_tensor

    def __len__(self):
        return len(self.data)
