import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import os
import numpy as np

class pixelclass_dataset_CIFAR10(torchvision.datasets.CIFAR10):
    """
    def __init__(self, rgb_image_paths):
        self.rgb_image_paths = rgb_image_paths  # Liste der Pfade zu RGB-Bildern
        #self.transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
    """
    def __init__(self, root, train=True, download=False, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    #def __len__(self):
     #   return len(self.rgb_image_paths)

    def __getitem__(self, idx):
        """
        rgb_image_path = self.rgb_image_paths[idx]
        rgb_image = Image.open(rgb_image_path).convert('RGB')
        """

        img, label = super().__getitem__(idx)

        label = torch.tensor(img)
        return img, label  # Eingabe und Ziel



"""
bild in lab umwandeln
pixelklassen erstellen als labels in form: tensor([3])  [3 ist die klasse]
"""