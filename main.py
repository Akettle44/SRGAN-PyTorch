import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from src.data.factory import TaskFactory
from src.data.imagenet import ImageNetDataset
from src.data.cifar10 import CIFAR10Dataset
from src.model.model import Generator, Discriminator
from src.train.train import PtTrainer
from src.model.load_save import saveModelToDisk, loadModelFromDisk
from src.utils.utils import Utils
from torchsummary import summary

# Trainer for models
def train():

    # Get configs
    root_dir = os.getcwd()
    dataset_name = "cifar"
    model_config_name = "srgan-small-standard"
    model_config, dataset_config = Utils.loadConfig(root_dir, model_config_name, dataset_name)

    # Create dataset
    dataset_dir = os.path.join(root_dir, dataset_config["dataset"]["path"])
    trbatch_sz = dataset_config["dataset"]["trbatch"]
    valbatch_sz = dataset_config["dataset"]["valbatch"]
    testbatch_sz = dataset_config["dataset"]["testbatch"]
    num_workers = dataset_config["dataset"]["numworkers"]
    blur_kernel_size = tuple(dataset_config["dataset"]["blur_kernel_size"])
    sigmas = tuple(dataset_config["dataset"]["sigmas"])
    train_val_test_split = tuple(dataset_config["dataset"]["train_val_test_split"])
    dataset = TaskFactory.createTaskDataSet(dataset_name, dataset_dir, blur_kernel_size, sigmas)

    # Create dataloaders
    train_val_test_split = [.7,.15,.15]
    # Seed split so that it is consistent across multiple runs
    train_dataset, val_dataset, test_dataset = random_split(dataset, train_val_test_split, generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=trbatch_sz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=valbatch_sz, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=testbatch_sz, shuffle=False, num_workers=num_workers)

    # Generator
    b1k_sz = model_config['model']['gen_block_1_kernel_size']
    b1p_sz = model_config['model']['gen_block_1_padding_size']
    n_resb = model_config['model']['gen_resid_blocks']
    cc = model_config['model']['conv_channels']
    scale = model_config['model']['scale_factor']
    g = Generator(b1k_sz, b1p_sz, n_resb, cc, scale)
    #summary(g, (3, 32, 32))

    # Disciminator
    dbs = model_config['model']['dis_blocks']
    dp = model_config['model']['dis_dropout']
    image_h = 32
    image_w = 32
    d = Discriminator(dbs, cc, dp, image_h, image_w)
    #summary(d, (3, 32, 32))
    
    #g, _ = loadModelFromDisk(root_dir, "srgan-small-standard", os.path.join(model_dir, model_save_name))

    # Train model
    loaders = [train_loader, val_loader, test_loader]
    trainer = PtTrainer(g, d, loaders, root_path=root_dir)
    trainer.sendToDevice()
    trainer.setHyps(hyps)
    trainer.updateOptimizerLr()
    tr_g, tr_d, val_g, val_d = trainer.fineTune()

    # Save model to disk
    model_dir = os.path.join(root_dir, 'models')
    num = len(os.listdir(model_dir)) + 1
    model_save_name = 'srgan-training' + '-' + str(num)
    saveModelToDisk(g, d, root_dir, model_save_name, model_config)

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


if __name__ == "__main__":
    train()
    #eval()
    #showSamples()