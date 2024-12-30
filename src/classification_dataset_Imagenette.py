import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import os
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import ImageFolder
"""
class PixelClassDataset_Imagenette(torch.utils.data.Dataset):
    #
    #custom Dataset for CIFAR10 for use in classification colorizing problem
    #
    
    def __init__(self, root, split,download, transform=None):
        self.data = torchvision.datasets.Imagenette(root=root, split=split, size="320px", download=download)
        self.transform = transform

    def __getitem__(self, index):
        img, _ = self.data[index]
        img_tensor = transforms.ToTensor()(img)  # Convert to tensor
        label_tensor = (img_tensor * 255).long()  # Create label tensor
        return img_tensor, label_tensor

    def __len__(self):
        return len(self.data)
"""

class PixelClassImagenette(torch.utils.data.Dataset):
    """
    Custom Dataset for Imagenette for use in pixel-level classification problem.
    """
    
    def __init__(self, root, train=True, transform=None, download=False):
        """
        Initialize the Imagenette dataset.
        
        Args:
        - root: Path to the dataset directory.
        - train: Boolean indicating whether to use the training or validation split.
        - transform: Optional transform to be applied on a sample.
        - download: Boolean to trigger the download of the dataset.
        """
        self.root = root
        self.split = 'train' if train else 'val'
        self.transform = transform or transforms.ToTensor()
        self.dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

        # Download and extract if necessary
        if download:
            self._download_and_extract()

        # Load dataset
        dataset_path = os.path.join(self.root, 'imagenette2-320', self.split)
        self.data = ImageFolder(dataset_path)

    def _download_and_extract(self):
        """
        Download and extract the Imagenette dataset if not already present.
        """
        archive_path = os.path.join(self.root, 'imagenette2-320.tgz')
        extracted_path = os.path.join(self.root, 'imagenette2-320')
        
        if not os.path.exists(extracted_path):
            print(f"Downloading Imagenette dataset to {archive_path}...")
            download_and_extract_archive(self.dataset_url, download_root=self.root)
            print("Download and extraction complete.")

    def __getitem__(self, index):
        """
        Fetch an item from the dataset.
        
        Args:
        - index: Index of the item to fetch.
        
        Returns:
        - img_tensor: Tensor representation of the image.
        - label_tensor: Tensor of pixel values (converted to long for classification).
        """
        img, _ = self.data[index]  # Load image and ignore label
        img_tensor = self.transform(img)  # Apply transformation (e.g., to Tensor)
        label_tensor = (img_tensor * 255).long()#.clamp(0, 255)  # wertebereich lag bei -255 bis 255, deswegen .clamp
        return img_tensor, label_tensor

    def __len__(self):
        """
        Return the size of the dataset.
        
        Returns:
        - Length of the dataset.
        """
        return len(self.data)