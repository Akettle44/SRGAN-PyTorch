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
            nn.Conv2d(3, 64, kernel_size=block_1_k_size, padding=block_1_padding),
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
        self.up1 = UpSample(conv_channels, scale)
        self.up2 = UpSample(conv_channels, scale)

        # Final conv to clean up upsampled feature representation
        self.convout = nn.Sequential(nn.Conv2d(conv_channels, 3, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.fextraction(x)
        out = self.resblocks(x)
        out = self.fconsolidation(out)
        
        # Element wise concat
        out = out + x

        out = self.up1(out)
        out = self.up2(out)
        out = self.convout(out)
        
        # Align with [-1, 1] input scale
        out = torch.tanh(out)
        return out

class DisBlock(nn.Module):
    '''
    Block for discriminator
    '''
    def __init__(self, channel, scale=1, stride=1):
        super(DisBlock, self).__init__()
        self.conv = nn.Conv2d(channel, channel * scale, kernel_size=3, padding=1, stride=stride)
        self.bn = nn.BatchNorm2d(channel * scale)
        self.lekrelu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lekrelu(x)

        return x

class Discriminator(nn.Module):
    '''
    SRGAN Discriminator
    '''
    def __init__(self, inp_h, inp_w):
        self.scaled_h = inp_h // 4
        self.scaled_w = inp_w // 4
        # Input height and width used to determine FC-size
        super(Discriminator, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        # k3n64s2
        self.block2 = DisBlock(64, 2, 2)
        # k3n128s1
        self.block3 = DisBlock(128, 1, 1)
        # k3n128s2
        self.block4 = DisBlock(128, 2, 2)
        # k3n256s1
        #self.block5 = DisBlock(256, 1, 1)
        # k3n256s2
        #self.block6 = DisBlock(256, 2, 2)
        # k3n512s1
        #self.block7 = DisBlock(512, 1, 1)
        # k3n512s2
        #self.block8= DisBlock(512, 1, 2)

        # FC layer mapping to prediction
        self.block9 = nn.Sequential(
            # Convs currently compress by 16, therefore scale 
            # 512 by that to compensate
            nn.Linear(256 * (self.scaled_h) * (self.scaled_w), 1024),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.5), # Extra noise during training
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        #x = self.block5(x)
        #x = self.block6(x)
        #x = self.block7(x)
        #x = self.block8(x)

        # Flatten everything after batch
        #x = torch.reshape(x , (x.shape[0], -1))
        #y = x.contiguous().view(x.size(0), -1)
        y = torch.flatten(x, start_dim=1)
        y = self.block9(y)
        y = torch.sigmoid(y.view(batch_size))

        return y