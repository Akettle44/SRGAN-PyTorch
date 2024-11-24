import os
import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.data.cifar10 import CIFAR10Dataset
from src.utils.img_processing import Downsample

@pytest.mark.usefixtures("setUp")
class TestCIFAR10Dataset():

    def testInit(self):
        """ Verify that the dataset initiates properly
        """
        root_dir = os.path.join(os.getcwd(), "datasets")
        blur_kernel_size = (5,9)
        sigma = (0.1,5.)
        batch_size = 32
        num_workers = 8

        cifar10 = CIFAR10Dataset(root_dir, blur_kernel_size, sigma, batch_size, num_workers)

        assert cifar10.root_dir == root_dir
        assert cifar10.blur_kernel_size == blur_kernel_size
        assert cifar10.sigma == sigma
        assert cifar10.batch_size == 32
        assert cifar10.num_workers == 8
        assert len(cifar10.dataset) == 60000
    
    def testDatasetSplits(self):
        """ Verifies that the dataset is correctly being
            split into training/validation/testing sets
        """
        root_dir = os.path.join(os.getcwd(), "datasets")
        blur_kernel_size = (5,9)
        sigma = (0.1,5.)
        batch_size = 32
        num_workers = 8

        cifar10 = CIFAR10Dataset(root_dir, blur_kernel_size, sigma, batch_size, num_workers)
        cifar10.createDataloaders(cifar10.batch_size, cifar10.num_workers, [0.5,0.3,0.2])

        assert len(cifar10.train_dataset) == 30000
        assert len(cifar10.val_dataset) == 18000
        assert len(cifar10.test_dataset) == 12000
        assert isinstance(cifar10.train_loader, DataLoader)
        assert isinstance(cifar10.val_loader, DataLoader)
        assert isinstance(cifar10.test_loader, DataLoader)
        
    def testDownsampling(self):
        """ Verifies that the dataset is correctly being
            downsampled using Gaussian blurring and bicubic 
            interpolation
        """
        root_dir = os.path.join(os.getcwd(), "datasets")
        blur_kernel_size = (5,9)
        sigma = (0.1,1.5)
        batch_size = 32
        num_workers = 8

        cifar10 = CIFAR10Dataset(root_dir, blur_kernel_size, sigma, batch_size, num_workers)
        image, label = cifar10.dataset[100]

        assert len(image[0]) == len(label[0])//4
        assert len(image[0][0]) == len(label[0][0])//4

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        denormalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std]
        )
        
        label = denormalize(label)

        transform = transforms.Compose([transforms.ToTensor(),
                                        Downsample(),
                                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                       ])
        
        to_pil = transforms.ToPILImage()
        non_gaussian = transform(to_pil(label))
        pixel_diff = int(torch.sum(torch.abs(torch.flatten(image) - torch.flatten(non_gaussian))))
        assert pixel_diff != 0