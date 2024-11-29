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


def show_images(trainer, loader):
    
    # Images, labels to device               
    images, labels = next(iter(loader))
    images_cuda = images.to('cuda')

    # Forward pass
    gens = trainer.generator(images_cuda)
    gens_detached = gens.detach() # Detach generations to prevent interference
    gens_detached = gens_detached.cpu()
    
    fig = plt.figure(figsize=(17,20), layout='compressed')
    batch_size = len(images)
    for i in range(batch_size):
        plt.subplot(batch_size, 3, 3*i+1)
        plt.imshow(images[i].permute(1, 2, 0))
        if i == 0: plt.title("LR")

        plt.subplot(batch_size, 3, 3*i+2)
        plt.imshow(labels[i].permute(1, 2, 0))
        if i == 0: plt.title("HR")

        plt.subplot(batch_size, 3, 3*i+3)
        plt.imshow(gens_detached[i].permute(1, 2, 0))
        if i == 0: plt.title("SR")
    
    plt.show()

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
    train_dataset, val_dataset, test_dataset = random_split(dataset, train_val_test_split)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers)

    # Model
    g = Generator(hyps['scale'])
    d = Discriminator(96, 96)
    model_name = f"SRGAN_epoch({hyps['epochs']})_scale({hyps['scale']})"
    loaders = [train_loader, val_loader, test_loader]
    trainer = PtTrainer(g, d, loaders)
    trainer.sendToDevice()
    trainer.setHyps(hyps)
    trainer.updateOptimizerLr()
    tr_g, tr_d, val_g, val_d = trainer.fineTune()
    saveModelToDisk(trainer.generator, trainer.discriminator, root_dir, model_name)
    
    # show example
    show_images(trainer, test_loader)

    # Plot loss results (show it decreases)
    save_path = os.path.join(root_dir, "models/" + model_name)
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

    # Plot loss results (show it decreases)
    """
    plt.figure()
    plt.plot(range(len(tr_acc)), tr_acc, label="Training Accuracy", color='blue')
    plt.plot(range(len(val_acc)), val_acc, label="Validation Accuracy", color='orange')
    plt.title(f"Training curves for {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(save_path, "accuracy_plot.png"))
    """

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
    train_dataset, val_dataset, test_dataset = random_split(dataset, train_val_test_split)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False, num_workers=num_workers)

    # Load Model
    g, d = loadModelFromDisk(specific_model, hyps)
    loaders = [test_loader, test_loader, test_loader]
    validator = PtTrainer(g, d, loaders)
    validator.sendToDevice()
    validator.setHyps(hyps)
    g_loss, d_loss, g_score, d_score = validator.test()
    print(f"Test g_loss = {g_loss}, d_loss = {d_loss}, g_score = {g_score}, d_score = {d_score}")

    # Try with a large image
    image_path = os.path.join(os.path.join(root_dir), "datasets\imagenet_test\cheetah\c_39.JPEG")
    image_HR = Image.open(image_path)
    image_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=sigma),
                                        Downsample(),
                                        #transforms.Resize(size=(24,24))
                                       ])
    image_LR = image_transform(image_HR)
    image_LR = image_LR.unsqueeze(0)
    image_LR_cuda = image_LR.to("cuda")
    image_SR_cude = validator.generator(image_LR_cuda)
    image_SR_cude = image_SR_cude.detach()
    image_SR = image_SR_cude.to("cpu")

    fig = plt.figure(figsize=(17,20))
    plt.subplot(1, 3, 1)
    plt.imshow(image_LR.squeeze(0).permute(1,2,0))
    plt.title("LR")

    plt.subplot(1, 3, 2)
    plt.imshow(image_SR.squeeze(0).permute(1,2,0))
    plt.title("SR")

    plt.subplot(1, 3, 3)
    plt.imshow(image_HR)
    plt.title("HR")

    plt.show()

    # # Dataset
    # blur_kernel_size = (5,9)
    # sigma = (0.1,1.5)
    # batch_size_train = hyps["trbatch"]
    # batch_size_val = hyps["valbatch"]
    # num_workers = hyps["numworkers"]

    # loadLocal = False
    # dataset = TaskFactory.createTaskDataSet(root_dir, blur_kernel_size, sigma, batch_size_train, num_workers)
    # dataset.createDataloaders(batch_size_train, num_workers)

    # # Define save directory for plots
    # base_save_dir = os.path.join(os.path.join(os.path.dirname(os.getcwd()), "plots"))
    # dirs = sorted(os.listdir(base_save_dir))
    # i = int(dirs[-1].split('_')[-1])
    # i += 1
    # run = "run_" + str(i)
    # save_dir = os.path.join(base_save_dir, run)
    # os.mkdir(save_dir)

    # # TODO: Evaluation

if __name__ == "__main__":
    #train()
    eval()