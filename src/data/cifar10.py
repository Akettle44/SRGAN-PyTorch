import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from overrides import override

from .dataset_interface import TaskDataset

class CIFAR10(Dataset):
    """ Modifies Torchvision's CIFAR10 dataset where:
            images: downsampled images
            labels: original images
    """
    def __init__(self, root, train=True, download=True, transform=None):
        self.cifar10 = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=None)
        self.image_transform = transform
        self.label_transform = transforms.Compose([transforms.ToTensor(), 
                                                   transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                   ])
    
    def __len__(self):
        return len(self.cifar10)
    
    def __getitem__(self, index):
        original_img, _ = self.cifar10[index]
        if self.image_transform:
            transformed_img = self.image_transform(original_img)
        else:
            transformed_img = self.label_transform(original_img)

        return transformed_img, self.label_transform(original_img)

class CIFAR10Dataset(TaskDataset):
    def __init__(self, root_dir, blur_kernel_size, sigma, size, batch_size=4, num_workers=2):
        super().__init__(root_dir, blur_kernel_size, sigma, size, batch_size, num_workers)
        self.load_dataset()

    @override
    def load_dataset(self):
        self.train_dataset = CIFAR10(root=self.root_dir,download=True,train=True,transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,
                                                       num_workers=self.num_workers)
        self.test_dataset = CIFAR10(root=self.root_dir,download=True, train=False, transform=self.transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,shuffle=False,
                                                       num_workers=self.num_workers)
    
