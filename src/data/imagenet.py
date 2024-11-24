# This file provides the class for ImageNetDataset

import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from overrides import override

from .dataset_interface import TaskDataset

class ImageNetDataset(Dataset, TaskDataset):

    """ Modifies ImageNet dataset where:
            images: downsampled images
            labels: original images
        Note: according to the paper, high-resolution images (labels) should be normalized to [-1,1],
        while low-resolution images remain in range [0,1]
    """

    @override
    def __init__(self, root_dir, blur_kernel_size, sigma, batch_size=32, num_workers=8, transform=None):
        # Rely on Python's MRO to do init
        super().__init__(root_dir, blur_kernel_size, sigma, batch_size, num_workers)

        self.class_dirs = root_dir
        self.imagenet = torchvision.datasets.ImageFolder(root=self.class_dirs)
        self.image_transform = self.transform
        self.label_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                ])
    def __len__(self):
        # Count the number of images in imagenet folder
        num_files = 0
        for _, _, files in os.walk(self.class_dirs):
            num_files += len(files)
        return num_files
        
    def __getitem__(self,index):
        original_img, _ = self.imagenet[index]
        if self.image_transform:
            transformed_img = self.image_transform(original_img)
        else:
            transformed_img = self.label_transform(original_img)
        return transformed_img, self.label_transform(original_img)