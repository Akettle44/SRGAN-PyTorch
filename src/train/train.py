### This file performs training in PyTorch

import torch
import os
from tqdm import tqdm
from src.model.loss import PerceptualLoss
from torchsummary import summary
from torch.profiler import profile, record_function, ProfilerActivity

class PtTrainer():

    def __init__(self, root_path, generator, discriminator, loaders, hyps):
        self.generator = generator
        self.discriminator = discriminator
        self.root_path = root_path
        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.test_loader = loaders[2]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyps = hyps

    def setupPreTraining(self):
        """ Set up pretraining using MSE only with generator
        """
        self.setOptimizer("pretraining")
        self.sendToDevice()
        criterian = torch.nn.MSELoss()
        g_sched = None
        if self.hyps['pre_g_sched']:
            match self.hyps['pre_g_sched']:
                case "plateau":
                    g_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer, 'min')
                case _:
                    raise NotImplementedError()
        return criterian, g_sched

    def setupTraining(self):
        """ Prepare for fine training or fine tuning
        """
        self.setOptimizer("training")
        self.sendToDevice()

        # Choose perceptual or MSE loss
        if self.hyps['loss'] == 'perceptual':
            vgg_path = os.path.join(self.root_path, self.hyps['loss_extractor'])
            loss = PerceptualLoss("perceptual", model_path=vgg_path)
        else:
            loss = PerceptualLoss("mse")

        g_sched = None
        # Learning rate scheduler generator
        if self.hyps['g_sched']:
            match self.hyps['g_sched']:
                case "plateau":
                    g_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer, 'min')
                case "multi":
                    g_sched = torch.optim.lr_scheduler.MultiStepLR(self.g_optimizer, milestones=[int(self.hyps['epochs'] / 2)])
                case _:
                    raise NotImplementedError()

        d_sched = None
        # Learning rate scheduler discriminator
        if self.hyps['d_sched']:
            match self.hyps['d_sched']:
                case "plateau":
                    d_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.d_optimizer, 'min')
                case "multi":
                    d_sched = torch.optim.lr_scheduler.MultiStepLR(self.d_optimizer, milestones=[int(self.hyps['epochs'] / 2)])
                case _:
                    raise NotImplementedError()

        return loss, g_sched, d_sched

    def setOptimizer(self, phase):
        """ Select appropriate opitimizer and associated params
        """
        match phase:
            case 'pretraining':
                # Generator
                match self.hyps['g_opt']:
                    case 'adam':
                        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.hyps['pre_g_lr'])
                    case _:
                        ValueError(f"Generator: Optimizer {self.hyps['g_opt']} isn't supported")
            case 'training':
                # Generator
                match self.hyps['g_opt']:
                    case 'adam':
                        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.hyps['g_lr'])
                    case _:
                        ValueError(f"Generator: Optimizer {self.hyps['g_opt']} isn't supported")
                
                # Discriminator
                match self.hyps['d_opt']:
                    case 'adam':
                        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.hyps['d_lr'])
                    case _:
                        ValueError(f"Discriminator: Optimizer {self.hyps['d_opt']} isn't supported")
            case _:
                raise ValueError(f"Phase {phase} not supported in GAN training")


    def updateOptimizerLr(self):
        """ Update the optimizers learning rate
        """
        # Generator
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = self.hyps['g_lr']

        # Discriminator
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = self.hyps['d_lr']

    def setHyps(self, hyps):
        """ Grab all hyperparameters and their associated value
        """
        for key, value in hyps.items():
            self.hyps[key] = value

    def setDevice(self, device):
        """ Updates the device  
        """
        self.device = device

    def sendToDevice(self):
        """ Places objects on correct device prior to training
        """
        if self.generator:
            self.generator = self.generator.to(self.device)
        if self.discriminator:
            self.discriminator = self.discriminator.to(self.device)

    def pretrain(self):
        """ Pretrain the generator using only MSE loss
        """
        
        criterian, g_sched = self.setupPreTraining()

        train_loss_g = []
        val_loss_g = []

        for epoch in range(self.hyps['pretrain_epochs']):

            ### TRAIN SINGLE EPOCH ###
            # Set training mode in PyTorch
            self.generator.train() 
            train_g_epoch_loss = []
            
            train_progress = tqdm(self.train_loader)
            # One iteration over dataset
            for batch in train_progress:

                # Images, labels to device               
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                gens = self.generator(images)
                loss = criterian(gens, labels)

                # Generator Loss Backprop
                self.g_optimizer.zero_grad()
                loss.backward()
                self.g_optimizer.step()

                # Training losses
                train_g_epoch_loss.append(loss.item())

            # Average loss and accuracy over epoch
            train_loss_g.append(torch.mean(torch.tensor(train_g_epoch_loss)))

            ### EPOCH VALIDATION ###
            # Set eval mode
            self.generator.eval() 

            val_g_epoch_loss = []

            with torch.no_grad():
                # Perform validation
                val_progress = tqdm(self.val_loader)
                for batch in val_progress:
                    # Images, labels to device               
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    gens = self.generator(images)
                    loss = criterian(gens, labels)

                    # Validation losses
                    val_g_epoch_loss.append(loss.item())

            # Average loss and accuracy over epoch
            val_loss_g.append(torch.mean(torch.tensor(val_g_epoch_loss)))
            ### END EPOCH VALIDATION ###

            # Results from each epoch of training
            print(f"Epoch {epoch+1}: train_loss_g: {train_loss_g[-1]}, val_loss_g: {val_loss_g[-1]}")
            g_sched.step(train_loss_g[-1])

        return train_loss_g,  val_loss_g

    def train(self):
        """ Train the model
        """        
        #torch.autograd.set_detect_anomaly(True)
        
        ### PROFILING ###
        #schedule = torch.profiler.schedule(
        #    wait=20,      # skip the first iteration
        #    warmup=5,    # warm up for one iteration
        #    active=10,     # only profile 5 iterations
        #    repeat=10,  # Repeat 10 times
        #    skip_first=True
        #)


        # Load optimizer + loss + schedulers
        loss, g_sched, d_sched = self.setupTraining()

        train_loss_g = []
        train_loss_d = []
        val_loss_g = []
        val_loss_d = []

        # Profile Training
        #with profile(activities=[ProfilerActivity.CUDA], schedule=schedule) as prof:
        #    with record_function("GAN_model_training"):

        for epoch in range(self.hyps['epochs']):

            ### TRAIN SINGLE EPOCH ###
            # Set training mode in PyTorch
            self.generator.train() 
            self.discriminator.train() 
            
            train_g_epoch_loss = []
            train_d_epoch_loss = []
            
            train_progress = tqdm(self.train_loader)
            # One iteration over dataset
            for batch in train_progress:

                # Images, labels to device               
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                ### UPDATE DISCRIMINATOR ###
                # Detach generations to prevent generator gradients from being updated
                gens = self.generator(images)
                gens_detached = gens.detach()
                d_fake = self.discriminator(gens_detached)
                d_real = self.discriminator(labels)

                # Compute discriminator loss
                d_loss, _ = loss(gens_detached, labels, d_fake, d_real, compute_d=True)

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()
                ### END UPDATE DISCRIMINATOR ###
               
                ### UPDATE GENERATOR ###
                d_fake_for_g = self.discriminator(gens) # Recompute 
                _, g_loss = loss(gens, labels, d_fake_for_g, None, compute_g=True)
                #print(g_loss)

                # Generator Loss Backprop
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                ### END UPDATE GENERATOR ###

                # Training losses
                train_g_epoch_loss.append(g_loss.item())
                train_d_epoch_loss.append(d_loss.item())

                #prof.step()

            # Average loss and accuracy over epoch
            train_loss_g.append(torch.mean(torch.tensor(train_g_epoch_loss)))
            train_loss_d.append(torch.mean(torch.tensor(train_d_epoch_loss)))

            # Print out per batch
            #train_progress.set_description(
            #    desc=f"epoch:[{epoch+1}], g_loss_train = {train_loss_g[-1]:.2f}, d_loss_train = {train_loss_d[-1]:.2f}")    
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
                for batch in val_progress:

                    # Images, labels to device               
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Detach generations to prevent generator gradients from being updated
                    gens = self.generator(images)
                    gens_detached = gens.detach()
                    d_fake = self.discriminator(gens_detached)
                    d_real = self.discriminator(labels)

                    # Compute discriminator loss
                    d_loss, _ = loss(gens_detached, labels, d_fake, d_real, compute_d=True)
                    d_fake_for_g = self.discriminator(gens) # Recompute 
                    _, g_loss = loss(gens, labels, d_fake_for_g, None, compute_g=True)

                    # Validation losses
                    val_g_epoch_loss.append(g_loss.item())
                    val_d_epoch_loss.append(d_loss.item())

                    # Print out per batch
                    #val_progress.set_description(
                    #desc=f"epoch:[{epoch+1}] \
                    #       g_loss_val = {val_g_epoch_loss[-1]:.2f}, \
                    #       d_loss_val = {val_d_epoch_loss[-1]:.2f}")    

            # Average loss and accuracy over epoch
            val_loss_g.append(torch.mean(torch.tensor(val_g_epoch_loss)))
            val_loss_d.append(torch.mean(torch.tensor(val_d_epoch_loss)))
            ### END EPOCH VALIDATION ###

            # Results from each epoch of training
            print(f"Epoch {epoch+1}: train_loss_g: {train_loss_g[-1]}, val_loss_g: {val_loss_g[-1]}")
            print(f"Epoch {epoch+1}: train_loss_d: {train_loss_d[-1]}, val_loss_d: {val_loss_d[-1]}")

            # Step learning rates if necessary
            if g_sched:
                g_sched.step()
            if d_sched:
                d_sched.step()

            # More profiling
            #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            #prof.export_chrome_trace("trace.json")

        return train_loss_g, train_loss_d, val_loss_g, val_loss_d

    def test(self):

        test_loss_g = []
        test_loss_d = []

        # Load optimizer + loss + schedulers
        loss, g_sched, d_sched = self.setupTraining()
        self.generator.eval() 
        self.discriminator.eval() 

        with torch.no_grad():
            test_progress = tqdm(self.test_loader)
            for batch in test_progress:

                # Images, labels to device               
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Detach generations to prevent generator gradients from being updated
                gens = self.generator(images)
                gens_detached = gens.detach()
                d_fake = self.discriminator(gens_detached)
                d_real = self.discriminator(labels)

                # Compute discriminator loss
                d_loss, _ = loss(gens_detached, labels, d_fake, d_real, compute_d=True)
                d_fake_for_g = self.discriminator(gens) # Recompute 
                _, g_loss = loss(gens, labels, d_fake_for_g, None, compute_g=True)

                # Test losses
                test_loss_g.append(g_loss.item())
                test_loss_d.append(d_loss.item())

        return torch.mean(torch.tensor(test_loss_g)), torch.mean(torch.tensor(test_loss_d))