### Loss Functions for SRGAN

import torch
import torchvision
import torch.nn.functional as F

# Perceptual Loss Function from SRGAN Paper
class PerceptualLoss(torch.nn.Module):
    def __init__(self, loss_choice, p_weight=10e-3, featureModel="vgg19", model_path=None):
        super(PerceptualLoss, self).__init__()
        self.loss_choice = loss_choice
        self.p_weight = p_weight
        if self.loss_choice == 'perceptual':
            self.featureNetwork = FeatureNetwork(featureModelChoice=featureModel, model_path=model_path)
        else:
            self.featureNetwork = None
        self.mse = torch.nn.MSELoss()

    def forward(self, hr_fake, hr_real, d_fake, d_real):
        """ Compute the perceptual loss for SRGAN

        Args:
            hr_fake (batch, image.dims): Predicted high resolution images
            hr_real (batch, image.dims): Groundtruth high resolution images

        Returns:
            d_loss, g_loss: Loss for discriminator and generator
        """
        g_loss = self.GLoss(hr_fake, hr_real, d_fake)
        #d_loss = self.DLoss_lsgan(d_fake, d_real)
        d_loss = self.DLoss(d_fake, d_real)

        return d_loss, g_loss

    def DLoss_lsgan(self, d_fake, d_real):
        l_real = 0.5 * torch.mean((d_real - 1)**2, axis=0)
        l_fake = 0.5 * torch.mean((d_fake)**2, axis=0)
        return l_real + l_fake
        
    def GLoss(self, hr_fake, hr_real, d_fake):
        """ Compute the Generator loss for SRGAN

        Args:
            hr_fake (torch.tensor): G(low_res)
            hr_real (torch.tensor): Labels
            d_fake (torch.tensor): D(G(low_res))

        Returns:
            g_loss: float
        """

        # Compute Content Loss
        match self.loss_choice:
            case "perceptual":
                # Features from fake images
                _ = self.featureNetwork(hr_fake)
                feat_fake = self.featureNetwork.features['feats']
                self.featureNetwork.clearFeatures()

                # Features from real images
                _ = self.featureNetwork(hr_real)
                feat_real = self.featureNetwork.features['feats']
                self.featureNetwork.clearFeatures()

                # MSE between representations (from paper)
                # 0.006 is the rescaling factor from the paper
                l_c = F.mse_loss(feat_fake, feat_real, reduction='mean') * 0.006 
            case "mse":
                l_c = F.mse_loss(hr_fake, hr_real, reduction='mean')
            case _:
                raise(NotImplementedError)

        # Compute adverserial loss
        l_a = -1 * torch.mean(torch.log(d_fake), dim=0)
        # Perform perceptual weighting
        g_loss = l_c + self.p_weight * l_a

        return g_loss

    def DLoss(self, d_fake, d_real):
        """ Compute the Discriminator Loss for SRGAN

        Args:
            hr_fake (torch.tensor): G(low_res)
            hr_real (torch.tensor): Labels
            d_fake (torch.tensor): D(G(low_res)) (after sigmoid)
            d_real (torch.tensor): D(real) (after sigmoid)

        Returns:
            d_loss: Discriminator loss
        """
        d_loss_real = F.binary_cross_entropy(d_real, torch.ones_like(d_real))
        d_loss_fake = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
        d_loss = d_loss_real + d_loss_fake 
        return d_loss

class FeatureNetwork(torch.nn.Module):
    """ Feature Network used for Perceptual Loss Function """
    def __init__(self, featureModelChoice, model_path=None):
        super(FeatureNetwork, self).__init__()
        # Select and load model
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selectPresetAndLoad(featureModelChoice)
        self.features = {}
        self.registerHooks()

    # Forward pass (ignore output, feature representation
    # auto populated from hook)
    def forward(self, x):
        x = self.model(x)
        return x 

    def clearFeatures(self):
        self.features = {}

    def selectPresetAndLoad(self, featureModelChoice):
        """ Choose a model and its preset
            preset: {name, layer_idx}
        """
        match featureModelChoice:
            case "vgg11":
                # 11 is after the 5th Conv / ReLU 
                # Got this number by printing out vgg.features and looking
                # at enumeration of sequential
                #self.preset = {"name": "vgg11", "layeridx": 12}
                self.preset = {"name": "vgg11", "layeridx": 4}

                # Load model from disk
                self.model = torch.load(self.model_path)
                self.model.classifier[6] = torch.nn.Linear(4096, 10) # Adjust for CIFAR

                #state_dict = torch.load(self.model_path).state_dict()
                #self.model = torchvision.models.vgg11(pretrained=False)
                #self.model.load_state_dict(state_dict)

                # Place on device in inference mode
                self.model = self.model.to(self.device)
                self.model.eval()
            case "vgg19":
                # 19 is after the 5th Conv / ReLU 
                # Got this number by printing out vgg.features and looking
                # at enumeration of sequential
                self.preset = {"name": "vgg19", "layeridx": 11}

                # Load model from disk
                self.model = torch.load(self.model_path)

                #state_dict = torch.load(self.model_path).state_dict()
                #self.model = torchvision.models.vgg19(pretrained=False)
                #self.model.load_state_dict(state_dict)

                # Place on device in inference mode
                self.model = self.model.to(self.device)
                self.model.eval()
            case _:
                raise(NotImplementedError)

    def registerHooks(self):
        """
        Register a hook to get the feature representation of a model 
        at a specififed point in the architecture
        NOTE: The hook is auto-populated during the forward pass. The output
        of the forward pass is NOT equivalent to the hook's variable
        """

        # Hook function (nested so that self can be used but isn't in function's signature)
        def hook_fn(module, input, output):
            self.features['feats'] = output.detach()

        # Register hook
        idx = self.preset["layeridx"]
        self.hook_handle = self.model.features[idx].register_forward_hook(hook_fn)

    """ 
    # Used if model ever needs to be directly modified

    def setFeatures(self):
        match self.preset.name:
            case "vgg19":
                self.features = torch.nn.Sequential(
                    self.model.features[0:self.preset["layeridx"]]
                )
            case _:
                raise(NotImplementedError)
    """