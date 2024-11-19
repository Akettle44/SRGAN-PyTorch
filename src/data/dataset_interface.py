import torchvision.transforms as transforms

from abc import ABC, abstractmethod

class TaskDataset(ABC):
    @abstractmethod
    def __init__(self,root_dir, blur_kernel_size, sigma, size, batch_size=32, num_workers=8):
        self.root_dir = root_dir
        self.blur_kernel_size = blur_kernel_size
        self.sigma = sigma
        self.size = size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.transform = transforms.Compose([transforms.GaussianBlur(kernel_size=self.blur_kernel_size, sigma=self.sigma),
                                        transforms.Resize(size=self.size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                       ])
        
    @abstractmethod
    def load_dataset(self):
        """ Loads training and testing datasets/loaders
        """
        pass
