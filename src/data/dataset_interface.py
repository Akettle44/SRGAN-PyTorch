# This file provides the parent class for Datasets
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
from torch.utils.data import random_split, DataLoader

from ..utils.img_processing import Downsample

class TaskDataset(ABC):
    @abstractmethod
    def __init__(self,root_dir, blur_kernel_size, sigma, batch_size=32, num_workers=8):
        """ blur_kernel_size: size of kernel used to downsample images
            sigma: standard deviation for creating kernel used to downsample images
        """
        self.root_dir = root_dir
        self.blur_kernel_size = blur_kernel_size
        self.sigma = sigma
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = None

        self.train_dataset = None
        self.train_loader = None
        self.val_dataset = None
        self.val_loader = None
        self.test_dataset = None
        self.test_loader = None

        self.downsample = transforms.Compose([transforms.ToTensor(),
                                        transforms.GaussianBlur(kernel_size=self.blur_kernel_size, sigma=self.sigma),
                                        Downsample(),
                                       ])
        
    def createDataloaders(self, batch_size, num_workers, train_val_test_split=[.7,.15,.15]):
        """ Splits dataset into training, validation, and testing splits
            train_val_test_split: allocate fraction of dataset to training/validation/testing dataloader
        """
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, train_val_test_split)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    