### Unit tests for SRGAN loss function ###

import torch
import torchvision
import os
import pytest

from src.model.loss import FeatureNetwork, PerceptualLoss

@pytest.mark.usefixtures("setUp")
class TestLoss():
    
    def test_FeatureNetInit(self, setUp):
        """ Test initialization and load model
        """

        fn = FeatureNetwork("vgg19")
        assert fn.features == {}

    def test_FeatureNetPopulation(self, setUp):
        """ Test variable population and load model
        """

        fn = FeatureNetwork("vgg19")
        assert fn.features == {}

        x = torch.randn(4, 3, 512, 512)
        fn.model(x)

        # 4 feature maps
        assert fn.features["vgg"].shape == (4, 256, 128, 128)

        # Test clearing
        fn.clearFeatures()
        assert fn.features == {}
                