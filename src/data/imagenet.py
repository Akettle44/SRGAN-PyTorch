# This file provides the class for ImageNetDataset

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from overrides import override

from .dataset_interface import TaskDataset

class ImageNetDataset(TaskDataset):
    def __init__(self, root_dir, blur_kernel_size, sigma, batch_size=32, num_workers=8):
        super().__init__(root_dir, blur_kernel_size, sigma, batch_size, num_workers)
        self.loadDataset()
    
    class ImageNet(Dataset):
        """ Modifies ImageNet dataset where:
                images: downsampled images
                labels: original images
            Note: according to the paper, high-resolution images (labels) should be normalized to [-1,1],
            while low-resolution images remain in range [0,1]
        """
        def __init__(self, root, transform=None):
            self.imagenet = torchvision.datasets.ImageFolder(root=root)
            self.image_transform = transform
            self.label_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                    ])
        def __len__(self):
            return len(self.imagenet)
        
        def __getitem__(self,index):
            original_img, _ = self.imagenet[index]
            if self.image_transform:
                transformed_img = self.image_transform(original_img)
            else:
                transformed_img = self.label_transform(original_img)
            return transformed_img, self.label_transform(original_img)

    @override
    def loadDataset(self):
        """ Load in images by placing them in SRGAN-PYTORCH/datasets/imagenet folder
        """
        self.dataset = self.ImageNet(self.root_dir,transform=self.transform)