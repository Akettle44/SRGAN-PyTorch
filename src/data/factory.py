# This file provides object instantiation for different datasets

from .cifar10 import CIFAR10Dataset
from .imagenet import ImageNetDataset

class TaskFactory():
    @staticmethod
    def createTaskDataSet(task_name, root_dir, blur_kernel_size, sigma, batch_size, num_workers):
        match task_name:
            case "CIFAR10":
                return CIFAR10Dataset(root_dir, blur_kernel_size, sigma, batch_size, num_workers)
            case "ImageNet":
                return ImageNetDataset(root_dir, blur_kernel_size, sigma, batch_size, num_workers)
            case _:
                raise ValueError(f"Task: {task_name} is not currently supported")