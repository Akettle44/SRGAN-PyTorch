# This file provides the class for CIFAR10Dataset
import torchvision
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, random_split
from overrides import override
from .dataset_interface import TaskDataset
from copy import deepcopy
from ..utils.img_processing import Downsample

class CIFAR10Dataset(Dataset):

    @override
    def __init__(self, root_dir, blur_kernel_size, sigma, batch_size=32, num_workers=8, train=True,
                            download=True):
        super().__init__()

        self.root_dir = root_dir
        self.blur_kernel_size = blur_kernel_size
        self.sigma = sigma
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = torchvision.datasets.CIFAR10(root=root_dir, train=train, download=download)
        # Perform gaussian blurring and downsampling (defined in dataset intereface because all methods use it)
        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.GaussianBlur(kernel_size=self.blur_kernel_size, sigma=self.sigma),
                                        Downsample(),
                                       ])
        #
        # Normalize to [-1, 1]
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

        # Create downsampled image
        ret_img = self.image_transform(img)

        # Normalize label image
        label = self.label_transform(img)

        return ret_img, label