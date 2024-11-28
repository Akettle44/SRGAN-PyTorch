import os
import torch
from .model import Generator, Discriminator
from .utils import Utils

def saveModelToDisk(generator, discriminator, root_dir, model_name):
    """ Save the GAN model to disk

    Args:
        generator (pytorch nn.module): generator
        discriminator (pytorch nn.module): discriminator
        model_dir (str): Directory to save model to
        run_number (int): Unique number to save the run files to
        model_name (str): Name for the model folder on disk
    """

    model_dir = os.path.join(os.path.join(root_dir, "models"), model_name)
    if( not os.path.exists(model_dir)):
        os.mkdir(model_dir)

    generator_dir = os.path.join(model_dir, "generator.pth")
    discriminator_dir = os.path.join(model_dir, "discriminator.pth")

    # Write the config, model, and tokenizer to disk
    torch.save(generator.state_dict(), generator_dir)
    torch.save(discriminator.state_dict(), discriminator_dir)

    # Write model metadata to disk
    # TODO: Make this relevant to GANs
    #metadata = os.path.join(model_dir, "metadata.txt")
    #with open(metadata, 'w') as f:
        #f.write('Legend: num_classes, task_type \n')
        #f.write(f"{model.num_classes}, {model.task_type}")

def loadModelFromDisk(model_dir):
    """ Load the model from disk locally

    Args:
        model_dir (str): path to model on disk
    """

    # Load Hyperparameters
    task_name = model_dir.split('/')[-1].split('-')[0]
    print(task_name)
    # Added 'models' to path to pass test case
    hyp_dir = os.path.join(os.path.dirname(os.path.dirname(model_dir)), 'models') # Insanely ratchet
    print(hyp_dir)
    hyps = Utils.loadHypsFromDisk(os.path.join(os.path.join(hyp_dir, 'hyps'), task_name + '.txt'))

    #metadata = os.path.join(model_dir, "metadata.txt")
    #with open(metadata, 'r') as f:
    #    for idx, line in enumerate(f):
    #        if idx == 1:
    #            num_classes, task_type = line.split(',')
    #            num_classes = int(num_classes)
    #            task_type = task_type.strip()

    # Load generator and discriminator
    gen_state_dict = torch.load(os.path.join(model_dir, 'generator.pth'))
    disc_state_dict = torch.load(os.path.join(model_dir, 'discriminator.pth'))

    g = Generator(scale=1) # Added scale=1 TODO: change to appropriate scale value
    g.load_state_dict(gen_state_dict)
    d = Discriminator()
    d.load_state_dict(disc_state_dict)

    return g, d, hyps
