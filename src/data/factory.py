# This file provides object instantiation for different datasets

from src.data.cifar10 import CIFAR10Dataset
from src.data.imagenet import ImageNetDataset

class TaskFactory():
    @staticmethod
    def createTaskDataSet(task_name, root_dir, sf, lr_proc, cropsz, blur_kernel_size, sigma):
        match task_name:
            case "cifar":
                #return CIFAR10Dataset(root_dir, blur_kernel_size, sigma, sf)
                pass
            case "imagenet":
                return ImageNetDataset(root_dir, sf)
            case _:
                raise ValueError(f"Task: {task_name} is not currently supported")