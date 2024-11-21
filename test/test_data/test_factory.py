import pytest
import os

from src.data.factory import TaskFactory
from src.data.cifar10 import CIFAR10Dataset
from src.data.imagenet import ImageNetDataset

@pytest.mark.usefixtures("setUp")
class TestTaskFactory():
    def testCIFAR10(self, setUp):
        """ Test that CIFAR10 object is instantiated correctly
        """
        task_name = "CIFAR10"
        root_dir = os.path.join(os.getcwd(), "datasets")
        blur_kernel_size = (5,9)
        sigma = (0.1, 5.)
        batch_size = 32
        num_workers = 8

        cifar10 = TaskFactory.createTaskDataSet(task_name,root_dir,blur_kernel_size,sigma,batch_size=batch_size,num_workers=num_workers)
        assert isinstance(cifar10, CIFAR10Dataset)

    def testImageNet(self, setUp):
        """ Test that ImageNet object is instantiated correctly
        """
        task_name = "ImageNet"
        root_dir = os.path.join(os.getcwd(), "datasets/imagenet")
        blur_kernel_size = (5,9)
        sigma = (0.1, 5.)
        batch_size = 32
        num_workers = 8

        imagenet = TaskFactory.createTaskDataSet(task_name,root_dir,blur_kernel_size,sigma,batch_size=batch_size,num_workers=num_workers)
        assert isinstance(imagenet, ImageNetDataset)