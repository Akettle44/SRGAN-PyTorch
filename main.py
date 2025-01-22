import torch
import argparse
import os
from torch.utils.data import random_split, DataLoader
from src.data.factory import TaskFactory
from src.model.model import Generator, Discriminator
from src.train.train import PtTrainer
from src.model.load_save import saveModelToDisk, loadModelFromDisk
from src.utils.utils import Utils
from torchsummary import summary

# Trainer for models
def train(root_dir, model_config_name, dataset_name, model_dir, save_name):

    ### CONFIGURATION ###
    model_config, dataset_config = Utils.loadConfig(root_dir, model_config_name, dataset_name)
    ### END CONFIGURATION ###
    
    ### MODEL ###
    # Generator
    b1k_sz = model_config['model']['gen_block_1_kernel_size']
    n_resb = model_config['model']['gen_resid_blocks']
    cc = model_config['model']['conv_channels']
    scale = model_config['model']['scale_factor']
    g = Generator(b1k_sz, n_resb, cc, scale)
    #summary(g, (3, 24, 24))

    # Disciminator
    dbs = model_config['model']['dis_blocks']
    dp = model_config['model']['dis_dropout']
    image_h, image_w = tuple(model_config['model']['crop_size'])
    d = Discriminator(dbs, cc, dp, image_h, image_w)
    #summary(d, (3, 96, 96))

    # Overwrite models by loading from disk
    if model_dir:
        g, d = loadModelFromDisk(root_dir, model_config_name, model_dir, image_h=image_h, image_w=image_w)
    ### END MODEL ###

    ### DATASET ###
    dataset_dir = os.path.join(root_dir, dataset_config["dataset"]["path"])
    trbatch_sz = dataset_config["dataset"]["trbatch"]
    valbatch_sz = dataset_config["dataset"]["valbatch"]
    testbatch_sz = dataset_config["dataset"]["testbatch"]
    num_workers = dataset_config["dataset"]["numworkers"]
    blur_kernel_size = tuple(dataset_config["dataset"]["blur_kernel_size"])
    sigmas = tuple(dataset_config["dataset"]["sigmas"])
    train_val_test_split = tuple(dataset_config["dataset"]["train_val_test_split"])

    # Create dataset
    if 'cifar' in dataset_name:
        dataset = TaskFactory.createTaskDataSet(dataset_name, dataset_dir, scale)
        train_val_test_split = [.7,.15,.15]
        # Seed split so that it is consistent across multiple runs
        train_dataset, val_dataset, test_dataset = random_split(dataset, train_val_test_split, generator=torch.Generator().manual_seed(42))
    elif 'imagenet' in dataset_name:
        train_dataset = TaskFactory.createTaskDataSet(dataset_name, os.path.join(dataset_dir, "subtrain"), scale)
        val_dataset = TaskFactory.createTaskDataSet(dataset_name, os.path.join(dataset_dir, "subval"), scale)
        test_dataset = TaskFactory.createTaskDataSet(dataset_name, os.path.join(dataset_dir, "subtest"), scale)
    else:
        raise ValueError(f"Dataset: {dataset_name} is not currently supported")

    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=trbatch_sz, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=valbatch_sz, shuffle=False, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=testbatch_sz, shuffle=False, num_workers=num_workers, drop_last=True)
    ### END DATASET ###

    ### TRAINING ###
    # Train model
    loaders = [train_loader, val_loader, test_loader]
    trainer = PtTrainer(root_dir, g, d, loaders, model_config['training'])
    trainer.sendToDevice()
    # Pretrain the generator using MSE only
    tr_g, val_g = trainer.pretrain()
    # Perform GAN training
    tr_g, tr_d, val_g, val_d = trainer.train()
    ### END TRAINING ###

    ### RECORD RESULTS ###
    # Save model to disk
    if not model_dir:
        model_dir = os.path.join(os.path.join(root_dir, 'models'), dataset_name)
    if not save_name:
        num = len(os.listdir(model_dir)) + 1
        save_name = 'srgan-training-run-' + str(num)
    save_path = os.path.join(model_dir, save_name)

    saveModelToDisk(g, d, root_dir, save_path, model_config)
    Utils.saveLosses(tr_g, tr_d, val_g, val_d, save_name, save_path)
    Utils.sampleModel(trainer.generator, test_loader, save_path)
    fid_score = Utils.computeFID(g, test_loader)
    Utils.saveFID(fid_score, save_path)
    ### END RECORD RESULTS ###

# Sampling from GAN
def evaluate(root_dir, model_config_name, dataset_name, model_dir):

    # Get configs
    model_config, dataset_config = Utils.loadConfig(root_dir, model_config_name, dataset_name, pretrained=True, model_loc=model_dir)

    # Generator
    b1k_sz = model_config['model']['gen_block_1_kernel_size']
    n_resb = model_config['model']['gen_resid_blocks']
    cc = model_config['model']['conv_channels']
    scale = model_config['model']['scale_factor']
    match dataset_name:
        case "imagenet":
            image_h, image_w = 256, 256 
        case "cifar":
            image_h, image_w = 32, 32
        case _:
            raise ValueError("Dataset name is invalid")
    g = Generator(b1k_sz, n_resb, cc, scale)
    g, _ = loadModelFromDisk(root_dir, model_config_name, model_dir, image_h=image_h, image_w=image_w)

    # Create dataset
    dataset_dir = os.path.join(root_dir, dataset_config["dataset"]["path"])
    testbatch_sz = dataset_config["dataset"]["testbatch"]
    num_workers = dataset_config["dataset"]["numworkers"]
    test_dataset = TaskFactory.createTaskDataSet(dataset_name, os.path.join(dataset_dir, "subtest"), scale, cropsz=(image_h, image_w))
    test_loader = DataLoader(test_dataset, batch_size=testbatch_sz, shuffle=False, num_workers=num_workers)

    # Record metrics
    #Utils.showSamples([g], [f'MSE Reconstructed: {tuple([image_h, image_w])}'], test_loader)
    Utils.sampleModel(g, test_loader, model_dir, "test_samples.png")
    fid = Utils.computeFID(g, test_loader)
    Utils.saveFID(fid, model_dir, "fid_test.txt")

if __name__ == "__main__":

    root_dir = os.getcwd()
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument('-m', '--mode', choices=['train', 'test'], help='Select whether to train or evaluate', required=True)
    # Optional
    parser.add_argument('-c', '--config', choices=['srgan-small-standard', 'srgan-medium-standard'], default='srgan-medium-standard', help='Select model size')
    parser.add_argument('-d', '--dataset', choices=['cifar', 'imagenet'], default='imagenet', help='Select dataset to use')
    parser.add_argument('-md', '--modeldir', type=str, help='Directory for model on disk, defaults to chosen dataset')
    parser.add_argument('-mn', '--modelname', type=str, help='Name of models directory, such as srgan-training-0')
    parser.add_argument('-lrm', '--lrmethod', type=str, choices=['bicubic', 'gaussian'], default='bicubic', help='Method to use for creating low resolution images during training')
    parser.add_argument('-s', '--savename', type=str, help='Name of saved model on disk')
    # Save name
    args = parser.parse_args()

    if args.mode == 'train':
        train(root_dir, args.config, args.dataset, args.modeldir, args.savename)
    else:
        model_dir = args.modeldir
        if args.modelname is None:
            parser.error('Model name is required for testing')
        elif model_dir is None:
            # If no model dir is provided, look in models/dataset for model. This is where training defaults to
            model_dir = os.path.join(os.path.join(os.path.join(root_dir, "models"), args.dataset), args.modelname)
        else:
            model_dir = os.path.join(os.path.join(root_dir, model_dir), args.modelname)
        # Assert that this location exists
        if not os.path.exists(model_dir):
            raise ValueError(f'Specified model directory: {model_dir} could not be found')

        evaluate(root_dir, args.config, args.dataset, model_dir) 