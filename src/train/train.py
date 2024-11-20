from model.model import Generator, Discriminator
import torch

# temporary
scale = 4
epoch = 0

def train():
    gen_net = Generator(4)  # scale factor of 4
    dis_net = Discriminator()

    # Saving the model parameter
    torch.save(gen_net.state_dict(), f"generator_scale_{scale}+epoch_{epoch}.pth")
    torch.save(dis_net.state_dict(), f"discriminator_scale_{scale}+epoch_{epoch}.pth")


