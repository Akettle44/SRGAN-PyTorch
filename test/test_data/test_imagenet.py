import os
import pytest
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import Tensor
from torch.utils.data import DataLoader, random_split
from src.data.imagenet import ImageNetDataset

@pytest.mark.usefixtures("setUp")
class TestImageNetDataset():

    def testInit(self):
        """ Verify that the dataset initiates properly
        """
        root_dir = os.path.join(os.getcwd(), "datasets/imagenet/Data/CLS-LOC")
        sf = 4
        lr_proc = "gaussian"
        cropsz = (128, 128)
        blur_kernel_size = (5,9)
        sigma = (0.1,0.5)

        imagenet = ImageNetDataset(root_dir, sf, lr_proc, cropsz, blur_kernel_size, sigma)

        assert imagenet.class_dir == root_dir
        assert imagenet.sf == sf
        assert imagenet.lr_proc == lr_proc
        assert imagenet.cropsz == cropsz
        assert imagenet.blur_kernel_size == blur_kernel_size
        assert imagenet.sigma == sigma

    def testDefaultInit(self):
        """ Verify that the dataset initiates properly
        """
        root_dir = os.path.join(os.getcwd(), "datasets/imagenet/Data/CLS-LOC")
        sf = 4

        imagenet = ImageNetDataset(root_dir, sf)
        assert imagenet.class_dir == root_dir
        assert imagenet.sf == sf
        assert imagenet.lr_proc == "bicubic"
        assert imagenet.cropsz == (96, 96)
        assert imagenet.blur_kernel_size == None
        assert imagenet.sigma == None

    def testImageNet(self):
        """ Verify that the dataset initiates properly
        """
        root_dir = os.path.join(os.getcwd(), "datasets/imagenet/Data/CLS-LOC")
        sf = 4

        imagenet = ImageNetDataset(root_dir, sf)
        lr, hr = imagenet[0]
        lshp = lr.shape
        hshp = hr.shape

        # Visual Check
        #torchvision.utils.save_image(lr, 'lr.png')
        #hr = (hr + 1) / 2
        #torchvision.utils.save_image(hr, 'hr.png')

        assert lshp[0] == 3
        assert lshp[1] == 24
        assert lshp[2] == 24

        assert hshp[0] == 3
        assert hshp[1] == 96
        assert hshp[2] == 96
        