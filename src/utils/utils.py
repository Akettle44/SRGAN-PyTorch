
import torch
import os
import yaml
import matplotlib.pyplot as plt
from torcheval.metrics import FrechetInceptionDistance

# Utility functions for bert pruning
class Utils():

    @staticmethod
    def loadConfig(root_dir, model_name, dataset_name, pretrained=False, model_loc=None):
        """ Load config files from disk

        Args:
            model_name: Model configuration (e.g. srgan-small)
            dataset_name: Dataset configuration (e.g. cifar)

        Returns:
            dictionaries with loaded configs
        """

        # Model
        if model_name:
            # Set path
            if pretrained:
                if not model_loc:
                    raise ValueError(f"Model path must be provided if loading from disk")
                model_path = os.path.join(model_loc, model_name + ".yaml")
            else:    
                model_path = os.path.join(os.path.join(os.path.join(root_dir, "configs"), "model"), model_name + ".yaml")

            # Verify path exists
            if not os.path.exists(model_path):
                raise ValueError(f"Model Configuration {model_name} does not exist")
            else:
                model_config = None
                with open(model_path, 'r') as f:
                    model_config = yaml.safe_load(f)
                if not model_config:
                    raise ValueError(f"Model Configuration {model_name} could not be loaded")

        # Dataset
        if dataset_name:
            dataset_path = os.path.join(os.path.join(os.path.join(root_dir, "configs"), "dataset"), dataset_name + ".yaml")
            if not os.path.exists(dataset_path):
                raise ValueError(f"Dataset Configuration {dataset_name} does not exist")
            else:
                dataset_config = None
                with open(dataset_path, 'r') as f:
                    dataset_config = yaml.safe_load(f)
                if not dataset_config:
                    raise ValueError(f"Dataset Configuration {dataset_name} could not be loaded")
        else:
            dataset_config = None
        
        return model_config, dataset_config

    @staticmethod
    def saveLosses(trl_g, trl_d, vl_g, vl_d, model_name, save_path):
        """ Save the loss plot to a figure

        Args:
            trl_g (torch.tensor): training loss generator
            trl_d (torch.tensor): training loss discriminator
            vl_g (torch.tensor):  validation loss generator
            vl_d (torch.tensor):  validation loss discriminator
        """
        # Plot loss results (show it decreases)
        plt.plot(range(len(trl_g)), trl_g, label="Training Loss Generator", color='blue')
        plt.plot(range(len(trl_d)), trl_d, label="Training Loss Discriminator", color='orange')
        plt.plot(range(len(vl_g)), vl_g, label="Validation Loss Generator", color='mediumseagreen')
        plt.plot(range(len(vl_d)), vl_d, label="Validation Loss Discriminator", color='crimson')
        plt.title(f"Training curves for {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(save_path, "loss_plot.png"))

    @staticmethod
    def sampleModel(generator, loader, save_path, save_name="samples.png"):

        generator.eval() 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Images, labels to device               
        images, labels = next(iter(loader))
        images_cuda = images.to(device)
        generator = generator.to(device)

        # Forward pass
        gens = generator(images_cuda)
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
            plt.savefig(os.path.join(save_path, save_name))

    @staticmethod
    def computeFID(generator, dloader):
        """ Compute the average Frechet Inception Distance (FID) score over 
            a test-set using a trained generator

        Args:
            generator (torch.nn.module): Trained generator
            dloader (torch.utils.data.dataloader): _description_
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set eval mode
        generator.eval() 
        fid = FrechetInceptionDistance(device=device)
        generator = generator.to(device)
        with torch.no_grad():
            for batch in dloader:
                # Images, labels to device               
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                gens = generator(images)

                # Convert real and fake from [-1, 1] to [0, 1]
                gens =   ((gens + 1) / 2).float()
                labels = ((labels + 1) / 2).float()

                # Update fake, update real
                fid.update(gens, False) # Fake
                fid.update(labels, True) # Real

        return fid.compute() 

    @staticmethod
    def saveFID(fid_score, save_path, file_name='fid_score.txt'):
        """ Write FID Score to Disk

        Args:
            fid_score (float)
            save_path (str): Path to directory
        """
        with open(os.path.join(save_path, file_name), 'w') as f:
            f.write(str(fid_score.item()))

    @staticmethod
    def showSamples(generators: list, gen_labels: list, dataloader):
        """ Show samples from different generators

        Args:
            generators (list): Different Generator models
            gen_labels (list): Label for each generator (e.g. MSE-Only)
            dataloader (torch.Dataloader): Where to import data from.
        """

        # Sample batch from dataloader
        it = iter(dataloader)
        lr, hr = next(it)
        
        # Grab samples from models
        samples = []
        for g in generators:
            ret = g(lr).detach()
            samples.append(ret)

        # Concatenate images together
        images = [lr, hr] + samples
        labels = [f'Low res: {tuple(lr.shape[2:])}', f'high res: {tuple(hr.shape[2:])}'] + gen_labels

        # Plot the concatenated images
        # Figure size: 3 (lr, hr, sr) x # of images in batch
        fig, axes = plt.subplots(dataloader.batch_size, len(labels), figsize=(8, 11))
        #fig.subplots_adjust(left=0.2, right=0.2, top=0.9, bottom=0.2)
        fig.tight_layout(pad=0.75, w_pad=0.1, h_pad=0.1)
        fig.patch.set_facecolor('#151b23') # Github dark mode background

        # Label axes
        for c, name in enumerate(labels):
            for r, image in enumerate(images[c]):
                # Set Title
                axes[0,c].set_title(name, color='white')

                # Isolate images, Convert (C, H, W) to (H, W, C)
                image = image.permute((1, 2, 0))
                if c != 0: # lr already in 0, 1
                    image = (((image + 1) / 2) * 255).byte()

                # Display image
                axes[r, c].imshow(image)
                axes[r, c].axis('off')

        plt.show()
    
