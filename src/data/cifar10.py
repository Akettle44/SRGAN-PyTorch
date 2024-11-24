# This file provides the class for CIFAR10Dataset
import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

from overrides import override
from .dataset_interface import TaskDataset
from copy import deepcopy

class CIFAR10Dataset(Dataset, TaskDataset):

    @override
    def __init__(self, root_dir, blur_kernel_size, sigma, batch_size=32, num_workers=8, train=True,
                            download=True, transform=None, do_transform=False):

        # Rely on Python's MRO to do intilialization
        super().__init__(root_dir, blur_kernel_size, sigma, batch_size, num_workers)

        self.dataset = torchvision.datasets.CIFAR10(root=root_dir, train=train, download=download, transform=None)
        self.do_transform = do_transform
        self.image_transform = transform
        self.label_transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    """ Modifies Torchvision's CIFAR10 dataset where:
            images: downsampled images
            labels: original images
        Note: according to the paper, high-resolution images (labels) should be normalized to [-1,1],
        while low-resolution images remain in range [0,1]
    """

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # Read in image (label not needed for our task)
        img, _ = self.dataset[index]

        # Perform image transform if necessary
        if self.do_transform:
            ret_img = self.image_transform(img)
        else:
            # Convert to tensor regardless
            ret_img = transforms.functional.pil_to_tensor(img)

        # Always perform label transform
        label = self.label_transform(img)

        return ret_img, label