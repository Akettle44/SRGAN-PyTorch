### This file performs training in PyTorch

import torch
from tqdm import tqdm
from src.model.loss import PerceptualLoss
from torchsummary import summary

class PtTrainer():

    def __init__(self, generator, discriminator, loaders, g_optimizer=None, d_optimizer=None):
        self.generator = generator
        self.discriminator = discriminator
        # TODO: Clean up
        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.test_loader = loaders[2]
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyps = {}

        if self.g_optimizer is None and self.d_optimizer is None:
            self.setDefaultOptimizer()

    def setDefaultOptimizer(self):
        """ Select appropriate opitimizer and associated params
        """
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=2e-5)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=2e-5)

    def updateOptimizerLr(self):
        """ Update the optimizers learning rate
        """
        # Generator
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = self.hyps['lr']

        # Discriminator
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = self.hyps['lr']

    def setDevice(self, device):
        """ Updates the device  
        """
        self.device = device

    def sendToDevice(self):
        """ Places objects on correct device prior to training
        """
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

    def setHyps(self, hyps):
        """ Grab all hyperparameters and their associated value
        """
        for key, value in hyps.items():
            self.hyps[key] = value

    def fineTune(self):
        
        # Loss
        loss = PerceptualLoss()

        #summary(self.generator, (3, 96, 96))
        #summary(self.discriminator, (3, 96, 96))

        # TODO: Accuracy not implemented yet
        train_loss = []
        train_accs = []
        val_loss = []
        val_accs = []

        for epoch in range(self.hyps['epochs']):

            ### TRAIN SINGLE EPOCH ###
            # Set training mode in PyTorch
            self.generator.train() 
            self.discriminator.train() 
            
            train_epoch_loss = []
            #train_batch_accs = []
            
            # One iteration over dataset
            for batch in tqdm(self.train_loader):

                # Images, labels to device               
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                gens = self.generator(images)
                d_fake = self.discriminator(gens)
                d_real = self.discriminator(labels)

                # Compute loss
                d_loss, g_loss = loss(gens, labels, d_fake, d_real)

                # Discriminator Loss
                self.d_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                self.d_optimizer.step()

                # Generator Loss
                self.g_optimizer.zero_grad()
                g_loss.backward(retain_graph=True)
                self.g_optimizer.step()

                # Training losses
                train_epoch_loss.append((g_loss.item(), d_loss.item()))

            # Average loss and accuracy over epoch
            train_loss.append(torch.mean(torch.tensor(train_epoch_loss)))
            #train_accs.append(torch.mean(torch.tensor(train_batch_accs)))
            ### END SINGLE EPOCH TRAIN ###

            ### EPOCH VALIDATION ###
            # Set eval mode
            self.generator.eval() 
            self.discriminator.eval() 

            val_epoch_loss = []
            #val_batch_accs = []

            with torch.no_grad():
                # Perform validation
                for batch in tqdm(self.val_loader):

                    # Images, labels                
                    images, labels = batch
                    images.to(self.device)
                    labels.to(self.device)

                    # Forward pass
                    gens = self.generator(images)
                    d_fake = self.discriminator(labels)
                    d_real = self.discriminator(gens)

                    # Compute loss
                    d_loss, g_loss = loss(gens, labels, d_fake, d_real)

                    # Validation losses
                    val_epoch_loss.append((g_loss.item(), d_loss.item()))
            
            val_loss.append(torch.mean(torch.tensor(val_epoch_loss)))
            # val_accs.append(torch.mean(torch.tensor(val_batch_accs)))
            ### END EPOCH VALIDATION ###

            # Results from each epoch of training
            print(f"Epoch {epoch}: train_loss: {train_loss[-1]}, val_loss: {val_loss[-1]}")
            #print(f"Epoch {epoch}: train_acc: {train_accs[-1]}, val_acc: {val_accs[-1]}")

        return train_loss, train_accs, val_loss, val_accs