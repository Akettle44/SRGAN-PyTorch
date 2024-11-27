import torch
from torch import nn


class ResidualBlock(nn.Module):
    '''
    This is Residuel Block in the Generator Network
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
        
        return x + output

class SubPixel(nn.Module):
    '''
    These are 2 subpixel blocks after skip connection in Generator
    '''
    def __init__(self, channel_in, scale):
        super(SubPixel, self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_in * scale, kernel_size=3, padding=1)
        self.pixshuff = nn.PixelShuffle(scale)
        self.perlu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixshuff(x)
        x = self.perlu(x)

        return x

class DisBlock(nn.Module):
    '''
    This is the main structure in Discriminator network
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

class Generator(nn.Module):
    def __init__(self, scale):
        super(Generator, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = ResidualBlock(64)
        self.block8 = ResidualBlock(64)
        self.block9 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        # do element-wise sum before passing into block 10 (in forward)

        self.block10 = SubPixel(64, scale)
        self.block11 = SubPixel(64, scale)
        self.block12 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=9, padding=4))

    def forward(self, x):
        x = self.block1(x)
        output = self.block2(x)
        output = self.block3(output)
        output = self.block4(output)
        output = self.block5(output)
        output = self.block6(output)
        output = self.block7(output)
        output = self.block8(output)
        output = self.block9(output)
        output = self.block10(output + x)
        output = self.block11(output)
        output = self.block12(output)

        #######################################
        # Maybe not just return output #
        #######################################
        return output




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        # k3n64s2
        self.block2 = DisBlock(64, 1, 2)
        # k3n128s1
        self.block3 = DisBlock(64, 2, 1)
        # k3n128s2
        self.block4 = DisBlock(128, 1, 2)
        # k3n256s1
        self.block5 = DisBlock(128, 2, 1)
        # k3n256s2
        self.block6 = DisBlock(256, 1, 2)
        # k3n512s1
        self.block7 = DisBlock(256, 2, 1)
        # k3n512s2
        self.block8= DisBlock(512, 1, 2)

        self.block9 = nn.Sequential(
            # Maybe this is wrong
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)

        return torch.sigmoid(x)