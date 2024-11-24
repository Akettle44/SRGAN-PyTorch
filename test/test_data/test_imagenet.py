import os
import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.data.imagenet import ImageNetDataset
from src.utils.img_processing import Downsample

@pytest.mark.usefixtures("setUp")
class TestImageNetDataset():
    def testInit(self):
        """ Verify that the dataset initiates properly
        """
        root_dir = os.path.join(os.getcwd(), "datasets/imagenet_test")
        blur_kernel_size = (5,9)
        sigma = (0.1,5.)
        batch_size = 32
        num_workers = 8

        imagenet = ImageNetDataset(root_dir, blur_kernel_size, sigma, batch_size, num_workers)

        assert imagenet.root_dir == root_dir
        assert imagenet.blur_kernel_size == blur_kernel_size
        assert imagenet.sigma == sigma
        assert imagenet.batch_size == 32
        assert imagenet.num_workers == 8
        assert len(imagenet.dataset) == 1300
    
    def testDatasetSplits(self):
        """ Verifies that the dataset is correctly being
            split into training/validation/testing sets
        """
        root_dir = os.path.join(os.getcwd(), "datasets/imagenet_test")
        blur_kernel_size = (5,9)
        sigma = (0.1,5.)
        batch_size = 32
        num_workers = 8

        imagenet = ImageNetDataset(root_dir, blur_kernel_size, sigma, batch_size, num_workers)
        imagenet.createDataloaders(imagenet.batch_size, imagenet.num_workers, [0.5,0.3,0.2])

        assert len(imagenet.train_dataset) == 650
        assert len(imagenet.val_dataset) == 390
        assert len(imagenet.test_dataset) == 260
        assert isinstance(imagenet.train_loader, DataLoader)
        assert isinstance(imagenet.val_loader, DataLoader)
        assert isinstance(imagenet.test_loader, DataLoader)

    def testDownsampling(self):
        """ Verifies that the dataset is correctly being
            downsampled using Gaussian blurring and bicubic 
            interpolation
        """
        root_dir = os.path.join(os.getcwd(), "datasets/imagenet_test")
        blur_kernel_size = (5,9)
        sigma = (0.1,1.5)
        batch_size = 4
        num_workers = 2

        imagenet = ImageNetDataset(root_dir, blur_kernel_size, sigma, batch_size, num_workers)
        image, label = imagenet.dataset[100]

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

    """ Use for visualizing images
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    denormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )

    label = denormalize(label)

    to_pil = transforms.ToPILImage()

    image_pil = to_pil(image)
    label_pil = to_pil(label)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image_pil)
    plt.title("Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(label_pil)
    plt.title("Label")
    plt.axis("off")

    plt.show()
    """