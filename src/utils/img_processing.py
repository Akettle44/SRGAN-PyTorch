# This file provides functions used for altering images
import torch
from torch.nn.functional import interpolate

class Downsample(torch.nn.Module):
    """ Downsample function performs bicubic interpolation
        on images
    """
    def __init__(self, sf):
        super().__init__()
        self.sf = sf

    def forward(self, img):
        batch = torch.unsqueeze(img,0)
        downsampled = interpolate(batch,
                                  scale_factor=1 / self.sf,
                                  recompute_scale_factor=True,
                                  mode='bicubic')
        result = torch.clamp(downsampled, min=0, max=255)
        return torch.squeeze(result)