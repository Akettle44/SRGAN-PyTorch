### Loss Functions for SRGAN

import torch
import torchvision
import torch.nn.functional as F

# Perceptual Loss Function from SRGAN Paper
class PerceptualLoss(torch.nn.Module):
    def __init___(self, p_weight=1e-3, featureModel="vgg19"):
        super(PerceptualLoss, self).__init__()
        self.p_weight = p_weight
        self.featureNetwork = FeatureNetwork(featureModelChoice=featureModel)

    def forward(self, hr_fake, hr_real, d_fake, d_real):
        """ Compute the perceptual loss for SRGAN

        Args:
            hr_fake (batch, image.dims): Predicted high resolution images
            hr_real (batch, image.dims): Groundtruth high resolution images

        Returns:
            d_loss, g_loss: Loss for discriminator and generator
        """

        g_loss = self.GLoss(hr_fake, hr_real, d_fake)
        d_loss = self.DLoss(d_fake, d_real)

        return d_loss, g_loss

    def GLoss(self, hr_fake, hr_real, d_fake, content_choice="feat"):
        """ Compute the Generator loss for SRGAN

        Args:
            hr_fake (torch.tensor): G(low_res)
            hr_real (torch.tensor): Labels
            d_fake (torch.tensor): D(G(low_res))
            content_choice (str, optional): Which content loss to use

        Returns:
            g_loss: float
        """

        # Compute Content Loss
        match content_choice:
            case "feat":
                # Features from fake images
                _ = self.featureNetwork(hr_fake)
                feat_fake = self.featureNetwork.features['vgg']
                self.featureNetwork.clearFeatures()

                # Features from real images
                _ = self.featureNetwork(hr_real)
                feat_real = self.featureNetwork.features['vgg']
                self.featureNetwork.clearFeatures()

                # MSE between representations (from paper)
                l_c = torch.nn.MSELoss(feat_fake, feat_real)

            case "mse":
                l_c = torch.nn.MSELoss(hr_fake, hr_real)
            case _:
                raise(NotImplementedError)

        # Compute adverserial loss
        l_a = -1 * torch.mean(F.logsigmoid(d_fake), dim=0)

        # Perform perceptual weighting
        g_loss = l_c + self.p_weight * l_a
        return g_loss

    def DLoss(self, d_fake, d_real):
        """ Compute the Discriminator Loss for SRGAN

        Args:
            hr_fake (torch.tensor): G(low_res)
            hr_real (torch.tensor): Labels
            d_fake (torch.tensor): D(G(low_res))
            d_real (torch.tensor): _description_

        Returns:
            d_loss: Discriminator loss
        """

        d_loss_real = torch.mean(F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)), dim=0)
        d_loss_fake = torch.mean(F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)), dim=0)
        d_loss = d_loss_real + d_loss_fake 
        return d_loss


class FeatureNetwork(torch.nn.Module):
    """ Feature Network used for Perceptual Loss Function """
    def __init__(self, featureModelChoice):
        super(FeatureNetwork, self).__init__()
        # Select and load model
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
            case "vgg19":
                # 11 is after the 5th Conv / ReLU 
                # Got this number by printing out vgg.features and looking
                # at enumeration of sequential
                self.preset = {"name": "vgg19", "layeridx": 11}
                self.model = torchvision.models.vgg19(pretrained=True)
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
            self.features['vgg'] = output.detach()

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