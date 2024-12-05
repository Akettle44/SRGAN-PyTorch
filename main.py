import torch
import numpy as np
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

    if save_path:
        plt.savefig(os.path.join(save_path, "test_set_plot.png"))

    plt.show()

# Trainer for models
def train():

    # Paths
    root_dir = os.getcwd()
    model_dir = os.path.join(root_dir, 'models')

    dataset_name = "CIFAR10"

    # Load Hyperparameters
    hyps = Utils.loadHypsFromDisk(os.path.join(os.path.join(model_dir, 'hyps'), dataset_name + '.txt'))

    # Dataset
    blur_kernel_size = (3,7)
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
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=num_workers)

    # Model
    #g = Generator(hyps['scale'])
    d = Discriminator(32, 32)
    model_name = f"SRGAN_epoch_{5}_scale_{hyps['scale']}_k3_23"
    specific_model = os.path.join(model_dir, model_name)
    g, _ = loadModelFromDisk(specific_model, hyps)
    loaders = [train_loader, val_loader, test_loader]
    trainer = PtTrainer(g, d, loaders, root_path=root_dir)
    trainer.sendToDevice()
    trainer.setHyps(hyps)
    trainer.updateOptimizerLr()
    tr_g, tr_d, val_g, val_d = trainer.fineTune()
    saveModelToDisk(trainer.generator, trainer.discriminator, root_dir, model_name + "_feat")
    save_path = os.path.join(root_dir, "models/" + model_name + "_feat")

    #saveModelToDisk(trainer.generator, trainer.discriminator, root_dir, model_name)
    #save_path = os.path.join(root_dir, "models/" + model_name)

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

    dataset_name = "CIFAR10"

    # Load Hyperparameters
    hyps = Utils.loadHypsFromDisk(os.path.join(os.path.join(model_dir, 'hyps'), dataset_name + '.txt'))

    # path
    model_name = f"SRGAN_epoch_{5}_scale_{hyps['scale']}_k3_19_feat"
    specific_model = os.path.join(model_dir, model_name)

    # Dataset
    blur_kernel_size = (3,7)
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
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=num_workers)

    # Load Model
    g, d = loadModelFromDisk(specific_model, hyps)
    loaders = [test_loader, test_loader, test_loader]
    validator = PtTrainer(g, d, loaders, root_path=root_dir)
    validator.sendToDevice()
    validator.setHyps(hyps)
    g_loss, d_loss, g_score, d_score = validator.test()
    print(f"Test g_loss = {g_loss}, d_loss = {d_loss}, g_score = {g_score}, d_score = {d_score}")

    # show example
    show_images(validator, test_loader, None)

def showSamples():
    """ Show samples from different parts of training
    """
    # Paths
    root_dir = os.getcwd()
    model_dir = os.path.join(root_dir, 'models')

    dataset_name = "CIFAR10"

    # Load Hyperparameters
    hyps = Utils.loadHypsFromDisk(os.path.join(os.path.join(model_dir, 'hyps'), dataset_name + '.txt'))

    # Dataset
    blur_kernel_size = (3,7)
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
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=num_workers)

    # Load Models
    mse_model = f"SRGAN_epoch_{5}_scale_{hyps['scale']}_k3_21"
    mse_path = os.path.join(model_dir, "cifar/generator_block12_k3/" + mse_model)
    gmse, _ = loadModelFromDisk(mse_path, hyps)

    feat_model = f"SRGAN_epoch_{5}_scale_{hyps['scale']}_k3_21_feat"
    feat_path = os.path.join(model_dir, "cifar/generator_block12_k3/" + feat_model)
    gfeat, _ = loadModelFromDisk(feat_path, hyps)

    distf_model = f"SRGAN_epoch_{5}_scale_{hyps['scale']}_k3_7_feat"
    distf_path = os.path.join(model_dir, "cifar/generator_block12_k3/" + distf_model)
    gdistf, _ = loadModelFromDisk(distf_path, hyps)

    lr, hr = next(iter(test_loader))
    mse_gens = gmse(lr).detach()
    feat_gens = gfeat(lr).detach()
    distf_gens = gdistf(lr).detach()

    #lr = (lr * 255).byte()
    # Concatenate images together
    images = [lr, hr, mse_gens, feat_gens, distf_gens]

    # Plotting the concatenated images
    fig, axes = plt.subplots(3, 5, figsize=(8, 5))
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    axes[0,0].set_title("LR")
    axes[0,1].set_title("HR")
    axes[0,2].set_title("MSE-Only")
    axes[0,3].set_title("Feat")
    axes[0,4].set_title("Dist Failure")
    fig.patch.set_facecolor('white')

    for i in range(lr.shape[0]):
        for j in range(len(images)):
            img = images[j][i].permute((1, 2, 0))  # Convert (C, H, W) to (H, W, C)
            # Normalize high resolution images from [-1, 1] to [0, 255]
            if j != 0:
                img = (((img + 1) / 2) * 255).byte()
            axes[i,j].imshow(img)
            axes[i,j].axis('on')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #train()
    #eval()
    showSamples()