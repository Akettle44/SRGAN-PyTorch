import os
import pytest
import torch
from src.data.cifar10 import CIFAR10Dataset
from torch.utils.data import DataLoader

@pytest.mark.usefixtures("setUp")
class TestCIFAR10Dataset():

    def testInit(self):
        root_dir = os.path.join(os.getcwd(), "datasets")
        blur_kernel_size = (5,9)
        sigma = (0.1,5.)
        size = 100
        batch_size = 32
        num_workers = 8

        cifar10 = CIFAR10Dataset(root_dir, blur_kernel_size, sigma, size, batch_size, num_workers)

        assert cifar10.root_dir == root_dir
        assert cifar10.blur_kernel_size == blur_kernel_size
        assert cifar10.sigma == sigma
        assert cifar10.size == size
        assert cifar10.batch_size == 32
        assert cifar10.num_workers == 8
    
    def testDatasetSplits(self):
        root_dir = os.path.join(os.getcwd(), "datasets")
        blur_kernel_size = (5,9)
        sigma = (0.1,5.)
        size = 100
        batch_size = 32
        num_workers = 8

        cifar10 = CIFAR10Dataset(root_dir, blur_kernel_size, sigma, size, batch_size, num_workers)
        cifar10.createDataloaders(cifar10.batch_size, cifar10.num_workers, [0.5,0.3,0.2])

        assert len(cifar10.train_dataset) == 30000
        assert len(cifar10.val_dataset) == 18000
        assert len(cifar10.test_dataset) == 12000
        assert isinstance(cifar10.train_loader, DataLoader)
        assert isinstance(cifar10.val_loader, DataLoader)
        assert isinstance(cifar10.test_loader, DataLoader)

    def testGaussian(self):
        root_dir = os.path.join(os.getcwd(), "datasets")
        blur_kernel_size = (5,9)
        sigma = (0.1, 1.5)
        size = 32

        cifar10 = CIFAR10Dataset(root_dir, blur_kernel_size, sigma, size)
        cifar10.createDataloaders(cifar10.batch_size, cifar10.num_workers, [0.5,0.3,0.2])
        image, label = cifar10.train_dataset[100]
        pixel_diff = int(torch.sum(torch.abs(torch.flatten(image) - torch.flatten(label))))
        assert pixel_diff < 500
        
    def testResizing(self):
        root_dir = os.path.join(os.getcwd(), "datasets")
        blur_kernel_size = (5,9)
        sigma = (0.1,1.5)
        size = 28
        batch_size = 4
        num_workers = 2

        cifar10 = CIFAR10Dataset(root_dir, blur_kernel_size, sigma, size, batch_size, num_workers)
        cifar10.createDataloaders(cifar10.batch_size, cifar10.num_workers, [0.5,0.3,0.2])
        image, label = cifar10.train_dataset[100]

        assert len(image[0]) == size
        assert len(image[0][0]) == size

        # import matplotlib.pyplot as plt
        # import torchvision.transforms as transforms
        # mean = [0.5, 0.5, 0.5]
        # std = [0.5, 0.5, 0.5]
        # denormalize = transforms.Normalize(
        #     mean=[-m / s for m, s in zip(mean, std)],
        #     std=[1 / s for s in std]
        # )
        # image = denormalize(image)
        # label = denormalize(label)

        # to_pil = transforms.ToPILImage()

        # image_pil = to_pil(image)
        # label_pil = to_pil(label)

        # plt.figure(figsize=(8, 4))

        # plt.subplot(1, 2, 1)
        # plt.imshow(image_pil)
        # plt.title("Image")
        # plt.axis("off")
        
        # plt.subplot(1, 2, 2)
        # plt.imshow(label_pil)
        # plt.title("Label")
        # plt.axis("off")

        # plt.show()