import pytest
import os
from src.data.factory import TaskFactory
from src.data.cifar10 import CIFAR10Dataset

@pytest.mark.usefixtures("setUp")
class TestTaskFactory():
    def testCIFAR10(self, setUp):
        """ Test that CIFAR10 object is instantiated correctly
        """
        task_name = "CIFAR10"
        root_dir = os.getcwd()
        blur_kernel_size = (5,9)
        sigma = (0.1, 5.)
        size = 100
        batch_size = 32
        num_workers = 8

        cifar10 = TaskFactory.createTaskDataSet(task_name,root_dir,blur_kernel_size,sigma,size,batch_size,num_workers)
        assert isinstance(cifar10, CIFAR10Dataset)