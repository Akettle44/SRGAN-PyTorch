### Unit tests for SRGAN loss function ###

import torch
import torchvision
import os
import torch.nn.functional as F
import pytest

from copy import deepcopy
from src.model.loss import FeatureNetwork, PerceptualLoss

@pytest.mark.usefixtures("setUp")
class TestLoss():
    
    def test_FeatureNetInit(self, setUp):
        """ Test initialization and load model
        """

        fn = FeatureNetwork("vgg19")
        assert fn.features == {}

    def test_FeaturePopulation(self, setUp):
        """ Test variable population and load model
        """

        fn = FeatureNetwork("vgg19")
        assert fn.features == {}

        x = torch.randn(4, 3, 512, 512)
        fn.model(x)

        # 4 feature maps
        assert fn.features["feats"].shape == (4, 256, 128, 128)

        # Test clearing
        fn.clearFeatures()
        assert fn.features == {}
                

    def test_PerceptualLossInit(self, setUp):
        """ Test loss function inititalization
        """

        loss = PerceptualLoss()
        assert loss.p_weight == 1e-3
        assert loss.featureNetwork.preset["name"] == "vgg19"

    def testPerceptualLossCalc(self, setUp):
        """ Test loss function initialization
        """

        loss = PerceptualLoss()
        img1 = torch.randn(1, 3, 512, 512)
        img2 = deepcopy(img1)

        d_fake = torch.zeros((1))
        d_real = torch.ones((1))

        d_l, g_l = loss(img1, img2, d_fake, d_real)

        assert torch.round(g_l) == 0
        assert d_l == 0


