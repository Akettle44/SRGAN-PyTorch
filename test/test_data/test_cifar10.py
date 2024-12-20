import os
import pytest
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from src.data.cifar10 import CIFAR10Dataset

@pytest.mark.usefixtures("setUp")
class TestCIFAR10Dataset():

    def test_Init(self):
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
        assert len(cifar10.dataset) == 50000
    
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
        # cifar10.createDataloaders(cifar10.batch_size, cifar10.num_workers)

        train_val_test_split = [0.7,0.15,0.15]
        train_dataset, val_dataset, test_dataset = random_split(cifar10, train_val_test_split)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # assert len(cifar10.train_dataset) == 35000
        # assert len(cifar10.val_dataset) == 7500
        # assert len(cifar10.test_dataset) == 7500
        assert len(train_dataset) == 35000
        assert len(val_dataset) == 7500
        assert len(test_dataset) == 7500
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

        # Tests to ensure that images loaded from dataset, dataloader are Tensors
        img_data, label_data = cifar10[0]
        img_trainload = None
        label_trainload = None
        img_valload = None
        label_valload = None
        img_testload = None
        label_testload = None
        for img, label in train_loader:
            img_trainload = img[0]
            label_trainload = label[0]
            break

        for img, label in val_loader:
            img_valload = img[0]
            label_valload = label[0]

        for img, label in test_loader:
            img_testload = img[0]
            label_testload = label[0]

        assert isinstance(img_data, Tensor)
        assert isinstance(label_data, Tensor)
        assert isinstance(img_trainload, Tensor)
        assert isinstance(label_trainload, Tensor)
        assert isinstance(img_valload, Tensor)
        assert isinstance(label_valload, Tensor)
        assert isinstance(img_testload, Tensor)
        assert isinstance(label_testload, Tensor)
        
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
        image, label = cifar10[100]

        # Verify downsampling occured
        _, iw, ih = image.shape
        _, lw, lh = label.shape
        assert iw == 96
        assert ih == 96

        # Image difference TODO: Make test empirical
        image = image * 255
        # For now, visually inspect image
        #plt.imshow(image.permute(1, 2, 0).numpy().astype('int'))
        #plt.show()

        # Assert label transform works properly
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        denormalize = transforms.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std]
        )
        # Reconstructed label
        recon_label = denormalize(label)
        label_diff = F.mse_loss(recon_label, label)
        eps = 1 # Arbitrary
        assert label_diff < eps