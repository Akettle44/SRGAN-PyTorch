# This file provides functions used for altering images
import torch
from torch.nn.functional import interpolate

class Downsample(torch.nn.Module):
    """ Downsample function performs bicubic interpolation
        on images
    """
    def forward(self, img):
        batch = torch.unsqueeze(img,0)
        downsampled = interpolate(batch,
                                  scale_factor=0.25,
                                  recompute_scale_factor=True,
                                  mode='bicubic')
        # Clamp removes noise in downsampled image
        result = torch.clamp(downsampled, min=0, max=255)
        return torch.squeeze(result)