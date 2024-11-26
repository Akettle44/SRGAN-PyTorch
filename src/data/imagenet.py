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
    def __init__(self, root_dir, blur_kernel_size, sigma, batch_size=32, num_workers=8):
        # Rely on Python's MRO to do init
        super().__init__(root_dir, blur_kernel_size, sigma, batch_size, num_workers)

        self.class_dirs = root_dir
        self.dataset = torchvision.datasets.ImageFolder(root=self.class_dirs)
        # Perform gaussian blurring and downsampling (defined in dataset intereface because all methods use it)
        self.image_transform = self.downsample
        # Normalize to [-1, 1]
        self.label_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                ])
    def __len__(self):
        # Count the number of images in imagenet folder
        return len(self.dataset)
        #num_files = 0
        #for _, _, files in os.walk(self.class_dirs):
        #    num_files += len(files)
        #return num_files
        
    def __getitem__(self, index):
        # Read in image (label not needed for our task)
        img, _ = self.dataset[index]

        # Create downsampled image
        ret_img = self.image_transform(img)

        # Normalize label image
        label = self.label_transform(img)

        return ret_img, label