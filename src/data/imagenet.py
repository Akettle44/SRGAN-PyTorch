# This file provides the class for ImageNetDataset

import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from overrides import override
from src.utils.img_processing import Downsample

class ImageNetDataset(Dataset):

    """ Modifies ImageNet dataset where:
            images: downsampled images
            labels: original images
        Note: according to the paper, high-resolution images (labels) should be normalized to [-1,1],
        while low-resolution images remain in range [0,1]
    """

    @override
    def __init__(self, root_dir, blur_kernel_size, sigma, batch_size=32, num_workers=8):
        
        super().__init__()

        self.root_dir = root_dir
        self.blur_kernel_size = blur_kernel_size
        self.sigma = sigma
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.class_dirs = root_dir
        self.dataset = torchvision.datasets.ImageFolder(root=self.class_dirs)
        # Perform gaussian blurring and downsampling (defined in dataset intereface because all methods use it)
        self.image_transform = transforms.Compose([
                                        transforms.GaussianBlur(kernel_size=self.blur_kernel_size, sigma=self.sigma),
                                        Downsample(),
                                        #transforms.Resize(size=(24,24))
                                       ])
        # Normalize to [-1, 1]
        self.label_transform = transforms.Compose([transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

        # Use before image_transform and label transform
        self.crop_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.RandomCrop(size=(96,96),pad_if_needed=True, padding=0)])
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

        # Perform random crop on images
        img_crop = self.crop_transform(img)

        # Create downsampled image
        ret_img = self.image_transform(img_crop)

        # Normalize label image
        #label = self.label_transform(img_crop)
        label = img_crop

        return ret_img, label