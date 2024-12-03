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
        
        #torch.autograd.set_detect_anomaly(True)

        # Loss
        loss = PerceptualLoss()

        # Learning rate scheduler generator
        g_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer, 'min')

        #summary(self.generator, (3, 8, 8))
        #summary(self.discriminator, (3, 32, 32))
        #exit(1)

        # TODO: Accuracy not implemented yet
        train_loss_g = []
        train_loss_d = []
        train_accs = []
        val_loss_g = []
        val_loss_d = []
        val_accs = []

        results = {"g_loss": [],
                   "d_loss": [],
                   "g_score": [],
                   "d_score": []
                   }

        for epoch in range(self.hyps['epochs']):

            ### TRAIN SINGLE EPOCH ###
            # Set training mode in PyTorch
            self.generator.train() 
            self.discriminator.train() 
            
            train_g_epoch_loss = []
            train_d_epoch_loss = []
            #train_batch_accs = []
            
            train_progress = tqdm(self.train_loader)
            running_results = {"g_loss": [],
                               "d_loss": [],
                               "g_score": [],
                               "d_score": []
                               }
        
            # One iteration over dataset
            for batch in train_progress:

                # Images, labels to device               
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                gens = self.generator(images)
                gens_detached = gens.detach() # Detach generations to prevent interference
                d_fake = self.discriminator(gens_detached)
                d_real = self.discriminator(labels)

                # Compute discriminator loss
                d_loss, _ = loss(gens_detached, labels, d_fake, d_real)

                # Discriminator Loss Backprop
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Compute discriminator loss for detached generator
                d_fake_for_g = self.discriminator(gens) # Recompute so tensors are different
                _, g_loss = loss(gens, labels, d_fake_for_g, torch.ones_like(d_real))

                # Generator Loss Backprop
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # store running result
                running_results["g_loss"].append(g_loss)
                running_results["d_loss"].append(d_loss)
                running_results["g_score"].append(torch.mean(d_real))
                running_results["d_score"].append(1 - torch.mean(d_fake_for_g))
                avg_g_loss = torch.mean(torch.tensor(running_results["g_loss"]))
                avg_d_loss = torch.mean(torch.tensor(running_results["d_loss"]))
                avg_g_score = torch.mean(torch.tensor(running_results["g_score"]))
                avg_d_score = torch.mean(torch.tensor(running_results["d_score"]))

                train_progress.set_description(desc="epoch:[%d/%d] g_loss = %.2f, d_loss = %.2f, g_score = %.2f, d_score = %.2f" % (
                epoch+1, self.hyps["epochs"],
                avg_g_loss,
                avg_d_loss,
                avg_g_score,
                avg_d_score
                ))
                results["g_loss"].append(avg_g_loss)
                results["d_loss"].append(avg_d_loss)
                results["g_score"].append(avg_g_score)
                results["d_score"].append(avg_d_score)

                # Training losses
                train_g_epoch_loss.append(g_loss.item())
                train_d_epoch_loss.append(d_loss.item())

            # Average loss and accuracy over epoch
            train_loss_g.append(torch.mean(torch.tensor(train_g_epoch_loss)))
            train_loss_d.append(torch.mean(torch.tensor(train_d_epoch_loss)))
            #train_accs.append(torch.mean(torch.tensor(train_batch_accs)))
            ### END SINGLE EPOCH TRAIN ###

            ### EPOCH VALIDATION ###
            # Set eval mode
            self.generator.eval() 
            self.discriminator.eval() 

            val_g_epoch_loss = []
            val_d_epoch_loss = []

            with torch.no_grad():
                # Perform validation
                val_progress = tqdm(self.val_loader)
                running_results = {"g_loss": [],
                                   "d_loss": [],
                                   "g_score": [],
                                   "d_score": []
                                   }
                
                for batch in val_progress:

                    # Images, labels to device               
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    gens = self.generator(images)
                    gens_detached = gens.detach() # Detach generations to prevent interference
                    d_fake = self.discriminator(gens_detached)
                    d_real = self.discriminator(labels)

                    # Compute discriminator loss
                    d_loss, _ = loss(gens_detached, labels, d_fake, d_real)

                    # Re-Compute discriminator loss for fake data
                    d_fake_for_g = self.discriminator(gens) # Recompute so tensors are different
                    _, g_loss = loss(gens, labels, d_fake_for_g, torch.ones_like(d_real))

                    running_results["g_loss"].append(g_loss)
                    running_results["d_loss"].append(d_loss)
                    running_results["g_score"].append(torch.mean(d_real))
                    running_results["d_score"].append(1 - torch.mean(d_fake_for_g))
                    avg_g_loss = torch.mean(torch.tensor(running_results["g_loss"]))
                    avg_d_loss = torch.mean(torch.tensor(running_results["d_loss"]))
                    avg_g_score = torch.mean(torch.tensor(running_results["g_score"]))
                    avg_d_score = torch.mean(torch.tensor(running_results["d_score"]))

                    val_progress.set_description(desc="Validation:[%d/%d] g_loss = %.2f, d_loss = %.2f, g_score = %.2f, d_score = %.2f" % (
                    epoch+1, self.hyps["epochs"],
                    avg_g_loss,
                    avg_d_loss,
                    avg_g_score,
                    avg_d_score
                    ))
                    
                    # Validation losses
                    val_g_epoch_loss.append(g_loss.item())
                    val_d_epoch_loss.append(d_loss.item())

            
            # Average loss and accuracy over epoch
            val_loss_g.append(torch.mean(torch.tensor(val_g_epoch_loss)))
            val_loss_d.append(torch.mean(torch.tensor(val_d_epoch_loss)))
            ### END EPOCH VALIDATION ###

            # Results from each epoch of training
            print(f"Epoch {epoch+1}: train_loss_g: {train_loss_g[-1]}, val_loss_g: {val_loss_g[-1]}")
            print(f"Epoch {epoch+1}: train_loss_d: {train_loss_d[-1]}, val_loss_d: {val_loss_d[-1]}")


            # Step learning rate if necessary
            g_sched.step(avg_g_loss)

        return train_loss_g, train_loss_d, val_loss_g, val_loss_d
    

    def test(self):
        # Loss
        loss = PerceptualLoss()
        results = {"g_loss": [],
                   "d_loss": [],
                   "g_score": [],
                   "d_score": []
                   }
        
        self.generator.eval() 
        self.discriminator.eval() 
        with torch.no_grad():
            test_progress = tqdm(self.test_loader)
            running_results = {"g_loss": [],
                    "d_loss": [],
                    "g_score": [],
                    "d_score": []
                    }

            for batch in test_progress:
                # Images, labels to device               
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                gens = self.generator(images)
                gens_detached = gens.detach() # Detach generations to prevent interference
                d_fake = self.discriminator(gens_detached)
                d_real = self.discriminator(labels)
                # Compute discriminator loss
                d_loss, g_loss = loss(gens, labels, d_fake, d_real)

                # store running result
                running_results["g_loss"].append(g_loss)
                running_results["d_loss"].append(d_loss)
                running_results["g_score"].append(torch.mean(d_real))
                running_results["d_score"].append(1 - torch.mean(d_fake))
                avg_g_loss = torch.mean(torch.tensor(running_results["g_loss"]))
                avg_d_loss = torch.mean(torch.tensor(running_results["d_loss"]))
                avg_g_score = torch.mean(torch.tensor(running_results["g_score"]))
                avg_d_score = torch.mean(torch.tensor(running_results["d_score"]))

                test_progress.set_description(desc="g_loss = %.2f, d_loss = %.2f, g_score = %.2f, d_score = %.2f" % (
                avg_g_loss,
                avg_d_loss,
                avg_g_score,
                avg_d_score
                ))

        return avg_g_loss, avg_d_loss, avg_g_score, avg_d_score