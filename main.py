import torch
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from src.data.imagenet import ImageNetDataset
from src.data.cifar10 import CIFAR10Dataset
from src.model.model import Generator, Discriminator
from src.train.train import PtTrainer
from src.model.load_save import saveModelToDisk, loadModelFromDisk
from src.utils.utils import Utils
from src.utils.img_processing import Downsample
from PIL import Image

def show_images(trainer, loader, save_path):
    
    # Images, labels to device               
    images, labels = next(iter(loader))
    images_cuda = images.to('cuda')

    # Forward pass
    gens = trainer.generator(images_cuda)
    gens = gens.detach().cpu()

    # Normalize from [0, 1] to [0, 255]
    images = (images * 255).byte()

    # Normalize high resolution images from [-1, 1] to [0, 255]
    gens = (((gens + 1) / 2) * 255).byte()
    labels = (((labels + 1) / 2) * 255).byte()
    
    fig = plt.figure(figsize=(10,12), layout='compressed')
    batch_size = len(images)
    for i in range(batch_size):
        plt.subplot(batch_size, 3, 3*i+1)
        plt.imshow(images[i].permute(1, 2, 0))
        if i == 0: plt.title("LR")

        plt.subplot(batch_size, 3, 3*i+3)
        plt.imshow(gens[i].permute(1, 2, 0))
        if i == 0: plt.title("SR")

        plt.subplot(batch_size, 3, 3*i+2)
        plt.imshow(labels[i].permute(1, 2, 0))
        if i == 0: plt.title("HR")
    plt.show()

    if save_path:
        plt.savefig
        plt.savefig(os.path.join(save_path, "test_set_plot.png"))

# Trainer for models
def train():

    # Paths
    root_dir = os.getcwd()
    model_dir = os.path.join(root_dir, 'models')

    dataset_name = "ImageNet"

    # Load Hyperparameters
    hyps = Utils.loadHypsFromDisk(os.path.join(os.path.join(model_dir, 'hyps'), dataset_name + '.txt'))

    # Dataset
    blur_kernel_size = (5,9)
    sigma = (0.1,1.5)
    batch_size_train = hyps["trbatch"]
    batch_size_val = hyps["valbatch"]
    num_workers = hyps["numworkers"]

    # Create dataset
    dataset_dir = os.path.join(os.path.join(root_dir, "datasets"), dataset_name.lower())
    if dataset_name == "ImageNet":
        dataset = ImageNetDataset(dataset_dir, blur_kernel_size, sigma, batch_size_train, num_workers)
    elif dataset_name == "CIFAR10":
        dataset = CIFAR10Dataset(dataset_dir, blur_kernel_size, sigma, batch_size_train, num_workers)

    # Create dataloaders
    train_val_test_split = [.7,.15,.15]
    # Seed split so that it is consistent across multiple runs
    train_dataset, val_dataset, test_dataset = random_split(dataset, train_val_test_split, generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers)

    # Model
    g = Generator(hyps['scale'])
    d = Discriminator(96, 96)
    model_name = f"SRGAN_epoch({hyps['epochs']})_scale({hyps['scale']})_4"
    loaders = [train_loader, val_loader, test_loader]
    trainer = PtTrainer(g, d, loaders)
    trainer.sendToDevice()
    trainer.setHyps(hyps)
    trainer.updateOptimizerLr()
    tr_g, tr_d, val_g, val_d = trainer.fineTune()
    saveModelToDisk(trainer.generator, trainer.discriminator, root_dir, model_name)
    
    save_path = os.path.join(root_dir, "models/" + model_name)

    # show example
    show_images(trainer, test_loader, save_path)

    # Plot loss results (show it decreases)
    plt.plot(range(len(tr_g)), tr_g, label="Training Loss Generator", color='blue')
    plt.plot(range(len(tr_d)), tr_d, label="Training Loss Discriminator", color='orange')
    plt.plot(range(len(val_g)), val_g, label="Validation Loss Generator", color='mediumseagreen')
    plt.plot(range(len(val_d)), val_d, label="Validation Loss Discriminator", color='crimson')
    plt.title(f"Training curves for {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_plot.png"))
    plt.show()

# Sampling from GAN
def eval():
    # Paths
    root_dir = os.getcwd()
    model_dir = os.path.join(root_dir, 'models')

    dataset_name = "ImageNet"

    # Load Hyperparameters
    hyps = Utils.loadHypsFromDisk(os.path.join(os.path.join(model_dir, 'hyps'), dataset_name + '.txt'))

    # path
    model_name = f"SRGAN_epoch({hyps['epochs']})_scale({hyps['scale']})"
    specific_model = os.path.join(model_dir, model_name)

    # Dataset
    blur_kernel_size = (5,9)
    sigma = (0.1,1.5)
    batch_size_train = hyps["trbatch"]
    batch_size_val = hyps["valbatch"]
    num_workers = hyps["numworkers"]

    # Create dataset
    dataset_dir = os.path.join(os.path.join(root_dir, "datasets"), dataset_name.lower())
    if dataset_name == "ImageNet":
        dataset = ImageNetDataset(dataset_dir, blur_kernel_size, sigma, batch_size_train, num_workers)
    elif dataset_name == "CIFAR10":
        dataset = CIFAR10Dataset(dataset_dir, blur_kernel_size, sigma, batch_size_train, num_workers)
    
    train_val_test_split = [.7,.15,.15]
    train_dataset, val_dataset, test_dataset = random_split(dataset, train_val_test_split, generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers)

    # Load Model
    g, d = loadModelFromDisk(specific_model, hyps)
    loaders = [test_loader, test_loader, test_loader]
    validator = PtTrainer(g, d, loaders)
    validator.sendToDevice()
    validator.setHyps(hyps)
    g_loss, d_loss, g_score, d_score = validator.test()
    print(f"Test g_loss = {g_loss}, d_loss = {d_loss}, g_score = {g_score}, d_score = {d_score}")

    # show example
    show_images(validator, test_loader, None)

if __name__ == "__main__":
    train()
    #eval()