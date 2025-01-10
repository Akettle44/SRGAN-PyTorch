# This file provides the class for ImageNetDataset
import torchvision
from torch.utils.data import Dataset
from overrides import override
from src.utils.image_processing import Preprocessing

class ImageNetDataset(Dataset):

    @override
    def __init__(self, root_dir, sf, lr_proc="bicubic", cropsz=(96, 96), blur_kernel_size=None, sigma=None):
        """ Initialize the ImageNetDataset Class
        Args:
            root_dir (str): Path to imagenet folder where classes live
            sf (int): Scale factor
            lr_proc (str): Process used to generate LR images. Options: [gaussian, bicubic]
            cropsz (tuple, optional): Size of random crop. Defaults to (96, 96).
            blur_kernel_size (2D-tuple): W,H for blurring kernel operation
            sigma (2D-tuple): Variances
        """

        super().__init__()
        self.class_dir = root_dir
        self.sf = sf
        self.lr_proc = lr_proc
        self.cropsz = cropsz

        # Gaussian Params
        self.blur_kernel_size = blur_kernel_size
        self.sigma = sigma

        # Dataset
        self.dataset = torchvision.datasets.ImageFolder(root=self.class_dir)

        # Transforms
        self.crop_transform = Preprocessing.createCropTransform(self.cropsz)
        self.lr_transform = Preprocessing.createLRTransform(self.lr_proc, self.sf, self.blur_kernel_size, self.sigma)
        self.hr_transform = Preprocessing.createHRTransform()

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        # Read in image (label not needed for our task)
        img, _ = self.dataset[index]

        # Perform random crop on image
        img_crop = self.crop_transform(img)

        # Create and scale lr and hr images
        lr = self.lr_transform(img_crop)
        hr = self.hr_transform(img_crop)

        return lr, hr