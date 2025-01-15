import os
import torch
import shutil
import yaml
from src.model.model import Generator, Discriminator
from src.utils.utils import Utils

def saveModelToDisk(generator, discriminator, root_dir, model_name, model_config):
    """ Save the GAN model to disk

    Args:
        generator (pytorch nn.module): generator
        discriminator (pytorch nn.module): discriminator
        model_dir (str): Directory to save model to
        run_number (int): Unique number to save the run files to
        model_name (str): Name for the model folder on disk
    """

    # Create save directory
    model_dir = os.path.join(os.path.join(root_dir, "models"), model_name)
    if( not os.path.exists(model_dir)):
        os.mkdir(model_dir)

    # Save generator and discriminator to disk
    generator_dir = os.path.join(model_dir, "generator.pth")
    discriminator_dir = os.path.join(model_dir, "discriminator.pth")
    torch.save(generator.state_dict(), generator_dir)
    torch.save(discriminator.state_dict(), discriminator_dir)

    # Save a copy of the current hyperparameter config
    with open(os.path.join(model_dir, model_config['model']['name'] + ".yaml"), 'w') as f:
        yaml.dump(model_config, f)

def loadModelFromDisk(root_dir, model_config_name, model_dir, \
                      loadD=False, image_h=None, image_w=None):
    """ Load the model from local disk

    Args:
        root_dir (str): root directory
        model_dir (str): path to model on disk
    """

    # Load model config
    model_config, _ = Utils.loadConfig(root_dir, model_config_name, None)
    
    # Generator
    gen_state_dict = torch.load(os.path.join(model_dir, 'generator.pth'))
    b1k_sz = model_config['model']['gen_block_1_kernel_size']
    n_resb = model_config['model']['gen_resid_blocks']
    cc = model_config['model']['conv_channels']
    scale = model_config['model']['scale_factor']
    g = Generator(b1k_sz, n_resb, cc, scale)
    g.load_state_dict(gen_state_dict)

    # Disciminator
    if loadD:
        disc_state_dict = torch.load(os.path.join(model_dir, 'discriminator.pth'))
        if not image_h or not image_w:
            ValueError("If you wish to load the discriminator, the image sizes must be provided")
        dbs = model_config['model']['dis_blocks']
        dp = model_config['model']['dis_dropout']
        d = Discriminator(dbs, cc, dp, image_h, image_w)
        d.load_state_dict(disc_state_dict)
    else:
        d = None

    return g, d
