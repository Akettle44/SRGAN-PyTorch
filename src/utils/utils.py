
import torch
import os
import yaml
import matplotlib.pyplot as plt

# Utility functions for bert pruning
class Utils():

    @staticmethod
    def loadConfig(root_dir, model_name, dataset_name):
        """ Load config files from disk

        Args:
            model_name: Model configuration (e.g. srgan-small)
            dataset_name: Dataset configuration (e.g. cifar)

        Returns:
            dictionaries with loaded configs
        """

        # Model
        model_path = os.path.join(os.path.join(os.path.join(root_dir, "configs"), "model"), model_name + ".yaml")
        if not os.path.exists(model_path):
            ValueError(f"Model Configuration {model_name} does not exist")
        else:
            model_config = None
            with open(model_path, 'r') as f:
                model_config = yaml.safe_load(f)
            if not model_config:
                ValueError(f"Model Configuration {model_name} could not be loaded")

        # Dataset
        dataset_path = os.path.join(os.path.join(os.path.join(root_dir, "configs"), "dataset"), dataset_name + ".yaml")
        if not os.path.exists(dataset_path):
            ValueError(f"Dataset Configuration {dataset_name} does not exist")
        else:
            dataset_config = None
            with open(dataset_path, 'r') as f:
                dataset_config = yaml.safe_load(f)
            if not dataset_config:
                ValueError(f"Dataset Configuration {dataset_name} could not be loaded")
        
        return model_config, dataset_config

    @staticmethod
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
    
    @staticmethod
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
