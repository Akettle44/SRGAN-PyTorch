import torchvision.transforms as transforms

from torch.utils.data import random_split, DataLoader

from abc import ABC, abstractmethod

class TaskDataset(ABC):
    @abstractmethod
    def __init__(self,root_dir, blur_kernel_size, sigma, size, batch_size=32, num_workers=8):
        """ blur_kernel_size: size of kernel used to downsample images
            sigma: standard deviation for creating kernel used to downsample images
            size: image's smallest edge matched to size
        """
        self.root_dir = root_dir
        self.blur_kernel_size = blur_kernel_size
        self.sigma = sigma
        self.size = size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = None

        self.train_dataset = None
        self.train_loader = None
        self.val_dataset = None
        self.val_loader = None
        self.test_dataset = None
        self.test_loader = None

        self.transform = transforms.Compose([transforms.GaussianBlur(kernel_size=self.blur_kernel_size, sigma=self.sigma),
                                        transforms.Resize(size=self.size),
                                        transforms.ToTensor(),
                                        # Normalizes pixels to be in range [-1, 1]
                                        # Supposed to help with model convergence, but can remove
                                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                       ])
        
    @abstractmethod
    def loadDataset(self):
        """ Loads whole dataset
        """
        pass

    def createDataloaders(self, batch_size, num_workers, train_val_test_split=[.5,.3,.2]):
        """ Splits dataset into training, validation, and testing splits
            train_val_test_split: allocate fraction of dataset to training/validation/testing dataloader
        """
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, train_val_test_split)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)