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

    # Generator
    b1k_sz = model_config['model']['gen_block_1_kernel_size']
    b1p_sz = model_config['model']['gen_block_1_padding_size']
    n_resb = model_config['model']['gen_resid_blocks']
    cc = model_config['model']['conv_channels']
    scale = model_config['model']['scale_factor']
    g = Generator(b1k_sz, b1p_sz, n_resb, cc, scale)
    #summary(g, (3, 4, 4))

    # Disciminator
    dbs = model_config['model']['dis_blocks']
    dp = model_config['model']['dis_dropout']
    image_h = 32
    image_w = 32
    d = Discriminator(dbs, cc, dp, image_h, image_w)
    #summary(d, (3, 32, 32))

    #g, _ = loadModelFromDisk(root_dir, "srgan-small-standard", os.path.join(model_dir, model_save_name))

    # Create dataset
    dataset_dir = os.path.join(root_dir, dataset_config["dataset"]["path"])
    trbatch_sz = dataset_config["dataset"]["trbatch"]
    valbatch_sz = dataset_config["dataset"]["valbatch"]
    testbatch_sz = dataset_config["dataset"]["testbatch"]
    num_workers = dataset_config["dataset"]["numworkers"]
    blur_kernel_size = tuple(dataset_config["dataset"]["blur_kernel_size"])
    sigmas = tuple(dataset_config["dataset"]["sigmas"])
    train_val_test_split = tuple(dataset_config["dataset"]["train_val_test_split"])
    dataset = TaskFactory.createTaskDataSet(dataset_name, dataset_dir, blur_kernel_size, sigmas, scale)

    # Create dataloaders
    train_val_test_split = [.7,.15,.15]
    # Seed split so that it is consistent across multiple runs
    train_dataset, val_dataset, test_dataset = random_split(dataset, train_val_test_split, generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=trbatch_sz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=valbatch_sz, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=testbatch_sz, shuffle=False, num_workers=num_workers)

    # Train model
    loaders = [train_loader, val_loader, test_loader]
    trainer = PtTrainer(root_dir, g, d, loaders, model_config['training'])
    trainer.sendToDevice()
    # Pretrain the generator using MSE only
    tr_g, val_g = trainer.pretrain()
    # Perform GAN training
    tr_g, tr_d, val_g, val_d = trainer.train()

    # Save model to disk + show examples from training
    model_dir = os.path.join(root_dir, 'models')
    num = len(os.listdir(model_dir)) + 1
    model_save_name = 'srgan-training' + '-' + str(num)
    saveModelToDisk(g, d, root_dir, model_save_name, model_config)
    save_path = os.path.join(model_dir, model_save_name)
    Utils.show_images(trainer, test_loader, save_path)

    # Plot loss results (show it decreases)
    plt.plot(range(len(tr_g)), tr_g, label="Training Loss Generator", color='blue')
    plt.plot(range(len(tr_d)), tr_d, label="Training Loss Discriminator", color='orange')
    plt.plot(range(len(val_g)), val_g, label="Validation Loss Generator", color='mediumseagreen')
    plt.plot(range(len(val_d)), val_d, label="Validation Loss Discriminator", color='crimson')
    plt.title(f"Training curves for {model_save_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_plot.png"))
    plt.show()

# Sampling from GAN
def eval():
    # Get configs
    root_dir = os.getcwd()
    dataset_name = "cifar"
    model_config_name = "SRGAN-small"
    model_disk_name= "srgan-training-15"
    model_dir = os.path.join(os.path.join(root_dir, "models"), model_disk_name)
    model_config, dataset_config = Utils.loadConfig(root_dir, model_config_name, dataset_name, pretrained=True, model_loc=model_dir)

    # Generator
    b1k_sz = model_config['model']['gen_block_1_kernel_size']
    b1p_sz = model_config['model']['gen_block_1_padding_size']
    n_resb = model_config['model']['gen_resid_blocks']
    cc = model_config['model']['conv_channels']
    scale = model_config['model']['scale_factor']
    g = Generator(b1k_sz, b1p_sz, n_resb, cc, scale)
    model_load_name = 'srgan-training-20'

    # Disciminator
    dbs = model_config['model']['dis_blocks']
    dp = model_config['model']['dis_dropout']
    image_h = 32
    image_w = 32
    d = Discriminator(dbs, cc, dp, image_h, image_w)

    g, _ = loadModelFromDisk(root_dir, "srgan-small-standard", model_dir, image_h=image_h, image_w=image_w)

    # Create dataset
    dataset_dir = os.path.join(root_dir, dataset_config["dataset"]["path"])
    trbatch_sz = dataset_config["dataset"]["trbatch"]
    valbatch_sz = dataset_config["dataset"]["valbatch"]
    testbatch_sz = dataset_config["dataset"]["testbatch"]
    num_workers = dataset_config["dataset"]["numworkers"]
    blur_kernel_size = tuple(dataset_config["dataset"]["blur_kernel_size"])
    sigmas = tuple(dataset_config["dataset"]["sigmas"])
    train_val_test_split = tuple(dataset_config["dataset"]["train_val_test_split"])
    dataset = TaskFactory.createTaskDataSet(dataset_name, dataset_dir, blur_kernel_size, sigmas, scale)

    # Create dataloaders
    train_val_test_split = [.7,.15,.15]
    # Seed split so that it is consistent across multiple runs
    train_dataset, val_dataset, test_dataset = random_split(dataset, train_val_test_split, generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=trbatch_sz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=valbatch_sz, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=testbatch_sz, shuffle=False, num_workers=num_workers)

    # Compute FID Score
    fid = Utils.computeFID(g, test_loader)
    print(fid)

if __name__ == "__main__":
    #train()
    eval()
    #showSamples()