import os
import pytest
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        assert len(imagenet) == 1300
    
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
        imagenet.createDataloaders(imagenet.batch_size, imagenet.num_workers, [0.7,0.15,0.15])

        assert len(imagenet.train_dataset) == 910
        assert len(imagenet.val_dataset) == 195
        assert len(imagenet.test_dataset) == 195
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
        image, label = imagenet[1]

        # Verify downsampling occured
        _, iw, ih = image.shape
        _, lw, lh = label.shape
        assert iw == lw // 4
        assert ih == lh // 4

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
        eps = 0.5 # Arbitrary
        assert label_diff < eps

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