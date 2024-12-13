import torch
import math
from torch import nn
import torchvision.transforms.functional as F

class ResidualBlock(nn.Module):
    '''
    This is Residuel Block in the generator network
    Note that spatial depth is maintained with padding, 
    this is essential for the residual layers
    '''
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.perlu = nn.PReLU()
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.perlu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = x + output
        
        return output

class UpSample(nn.Module):
    '''
    Upsample the image spatially
    For scale factor = r, conv (c_in, c_out r**2)
    Reshape output of conv to shape: (c_in, h * r, w * r)
    '''
    def __init__(self, channel_in, scale):
        super(UpSample, self).__init__()
        self.up_scale = int(math.sqrt(scale))
        self.conv = nn.Conv2d(channel_in, channel_in * self.up_scale**2, kernel_size=3, padding=1)
        self.pixshuff = nn.PixelShuffle(self.up_scale)
        self.perlu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixshuff(x)
        x = self.perlu(x)
        return x

class Generator(nn.Module):
    '''
    SRGAN Generator
    '''
    def __init__(self, block_1_k_size, block_1_padding, num_resid_blocks, conv_channels, \
                 scale):
        super(Generator, self).__init__()

        # Initial feature extraction
        # Kernel size should be tuned to size of images
        self.fextraction = nn.Sequential(
            nn.Conv2d(3, conv_channels, kernel_size=block_1_k_size, padding=block_1_padding),
            nn.PReLU()
        )

        # Stack residual block
        # The number of residual blocks is the main difference across network sizes
        self.resblocks = nn.Sequential(*[ResidualBlock(conv_channels) for i in range(num_resid_blocks)])

        # Consolidate features from residual extraction
        self.fconsolidation = nn.Sequential(
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels)
        )

        # Upsample to HR image size
        self.uplayers = nn.Sequential(*[UpSample(conv_channels, scale) for i in range(int(math.log2(scale)))])

        # Final conv to clean up upsampled feature representation
        self.convout = nn.Sequential(nn.Conv2d(conv_channels, 3, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.fextraction(x)
        out = self.resblocks(x)
        out = self.fconsolidation(out)
        
        # Element wise concat
        out = out + x

        out = self.uplayers(out)
        out = self.convout(out)
        
        # Align with [-1, 1] input scale
        out = torch.tanh(out)
        return out

class DisBlock(nn.Module):
    '''
    Block for discriminator
    Each block increases the depth by 2 and the spatial dimensions by 2
    '''
    def __init__(self, cc):
        super(DisBlock, self).__init__()
        self.dblock = nn.Sequential(
            nn.Conv2d(cc, cc, kernel_size=3, padding=1),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cc, cc * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(cc * 2),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.dblock(x)

class Discriminator(nn.Module):
    '''
    SRGAN Discriminator
    '''
    def __init__(self, nblocks, cc, dropout, inp_h, inp_w):
        super(Discriminator, self).__init__()

        # Feature extractor
        self.fextraction = nn.Sequential(
            nn.Conv2d(3, cc, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(cc, cc * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(cc * 2),
            nn.LeakyReLU(0.2)
        )

        # Progressively increase depth and reduce spatial dimensions
        # e.g. if cc = 64, nblocks = 3, depths: 64 (fextraction), 128, 256
        self.dblocks = nn.Sequential(*[DisBlock(cc*(2**i)) for i in range(1, nblocks)])

        # FC layer mapping to prediction
        change = 2**nblocks
        self.cls = nn.Sequential(
            # Depth given by cc * (2**nblocks)
            # Spatial: inp_h // (2**nblocks), inp_w // (2**nblocks)
            nn.Linear(cc * (change) * \
                     (inp_h // change) * (inp_w // change), \
                     1024),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(dropout), # Extra noise during training
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.fextraction(x)
        out = self.dblocks(out)

        out = torch.flatten(out, start_dim=1)
        out = self.cls(out)
        out = torch.sigmoid(out.view(batch_size))

        return out