# This file provides object instantiation for different datasets

from src.data.cifar10 import CIFAR10Dataset
from src.data.imagenet import ImageNetDataset

class TaskFactory():
    @staticmethod
    def createTaskDataSet(task_name, root_dir, sf, **kwargs):
        match task_name:
            case "cifar":
                #return CIFAR10Dataset(root_dir, blur_kernel_size, sigma, sf)
                pass
            case "imagenet":
                # Set defaults here
                lr_proc = kwargs.get('lr_proc', "bicubic")
                cropsz = kwargs.get('cropsz', (96, 96))
                blur_kernel_size = kwargs.get('blur_kernel_size', None)
                sigma = kwargs.get('sigma', None)

                return ImageNetDataset(root_dir, sf, lr_proc=lr_proc, cropsz=cropsz, \
                                       blur_kernel_size=blur_kernel_size, sigma=sigma)
            case _:
                raise ValueError(f"Task: {task_name} is not currently supported")