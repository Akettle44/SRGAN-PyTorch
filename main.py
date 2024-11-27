import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from src.data.imagenet import ImageNetDataset
from src.data.cifar10 import CIFAR10Dataset
from src.model.model import Generator, Discriminator
from src.train.train import PtTrainer
from src.model.load_save import saveModelToDisk, loadModelFromDisk
from src.utils.utils import Utils

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

    """
    # Model
    g = Generator(hyps['scale'])
    d = Discriminator()
    model_name = 'gan-01'
    
    trainer = PtTrainer(g, d, dataset)
    trainer.sendToDevice()
    trainer.setHyps(hyps)
    trainer.updateOptimizerLr()
    tr_loss, tr_acc, val_loss, val_acc = trainer.fineTune()
    saveModelToDisk(trainer.generator, trainer.discriminator, root_dir, model_name)
    
    # Plot loss results (show it decreases)
    save_path = os.path.join(root_dir, "models/" + model_name)
    plt.plot(range(len(tr_loss)), tr_loss, label="Training Loss", color='blue')
    plt.plot(range(len(val_loss)), val_loss, label="Validation Loss", color='orange')
    plt.title(f"Training curves for {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_plot.png"))
    """

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
    model_name = "gan-01"
    root_dir = os.path.dirname(os.getcwd())
    model_dir = os.path.join(root_dir, "models")
    specific_model = os.path.join(model_dir, model_name)

    # Load Model
    g, d, hyps = loadModelFromDisk(specific_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    blur_kernel_size = (5,9)
    sigma = (0.1,1.5)
    batch_size_train = hyps["trbatch"]
    batch_size_val = hyps["valbatch"]
    num_workers = hyps["numworkers"]

    loadLocal = False
    dataset = TaskFactory.createTaskDataSet(root_dir, blur_kernel_size, sigma, batch_size_train, num_workers)
    dataset.createDataloaders(batch_size_train, num_workers)

    # Define save directory for plots
    base_save_dir = os.path.join(os.path.join(os.path.dirname(os.getcwd()), "plots"))
    dirs = sorted(os.listdir(base_save_dir))
    i = int(dirs[-1].split('_')[-1])
    i += 1
    run = "run_" + str(i)
    save_dir = os.path.join(base_save_dir, run)
    os.mkdir(save_dir)

    # TODO: Evaluation

if __name__ == "__main__":
    train()
    #eval()