# This file provides functions used for altering images

import torch
import torchvision.transforms as transforms
from torch.nn.functional import interpolate

class Preprocessing:
    """ Collection for all of the preprocessing transforms
    """

    class Bicubic(torch.nn.Module):
        """ Downsample function performs bicubic interpolation
            on images
        """
        def __init__(self, sf):
            super().__init__()
            self.sf = sf

        def forward(self, hr):
            batch = torch.unsqueeze(hr, 0)
            downsampled = interpolate(batch,
                                    scale_factor= 1 / self.sf,
                                    recompute_scale_factor=False,
                                    mode='bicubic')

            return torch.squeeze(torch.clamp(downsampled, 0, 1))

    # Used by LR and HR
    @staticmethod
    def createCropTransform(cropsz):
        """ 
        Create the crop transform used by datasets

        Args:
            cropsz (2D Tuple): W,H for cropping
        """

        # Transforms
        crop_transform = transforms.Compose([transforms.ToTensor(), # Zero pad if necessary
                                                transforms.RandomCrop(size=(cropsz[0],cropsz[1]), pad_if_needed=True, padding=0)])
        return crop_transform
    
    @staticmethod
    def createLRTransform(method, sf, blur_kernel_size, sigma):
        """
        Create the LR transform used by datasets

        Args:
            method(str): Choice of LR method
            sf (int): Scale factor
            blur_kernel_size (2D-tuple): W,H for blurring kernel operation
            sigma (2D-tuple): Variances
        """

        match method:
            case "bicubic":
                pass
                t = transforms.Compose([Preprocessing.Bicubic(sf)])
            case "gaussian":
                t = transforms.Compose([transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=sigma), 
                                        Preprocessing.Bicubic(sf)])
            case _:
                raise NotImplementedError("Preprocessing: Not a valid LR method")

        return t
    
    @staticmethod
    def createHRTransform():
        """
        Create HR transform uses by datasets
        """
        # Normalize to [-1, 1]
        hr_transform = transforms.Compose([transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        return hr_transform

