## Unit tests for SRGAN save/loss functions

import torch
import os
import torch.nn.functional as F
import pytest

from src.model.load_save import saveModelToDisk, loadModelFromDisk
from src.model.model import Generator, Discriminator

@pytest.mark.usefixtures("setUp")
class TestSaveLoad():
    def test_SaveModel(self):
        """ Verify that the model saves properly
        """
        generator = Generator(scale=1)
        discriminator = Discriminator()
        root_dir = os.getcwd()
        saveModelToDisk(generator, discriminator, root_dir, "unit_test")

        assert os.path.exists(os.path.join(root_dir, os.path.join("models/pytest", "generator.pth")))
        assert os.path.exists(os.path.join(root_dir, os.path.join("models/pytest", "discriminator.pth")))

    def test_LoadModel(self):
        """ Verify that saved models load properly
        """
        root_dir = os.getcwd()
        g_orig = Generator(scale=1)
        d_orig = Discriminator()
        model_dir = os.path.join(root_dir, "models/unit_test")
        saveModelToDisk(g_orig, d_orig, root_dir, "unit_test")
        
        g_saved, d_saved, h_saved = loadModelFromDisk(model_dir)
        g_orig_state = g_orig.state_dict()
        g_saved_state = g_saved.state_dict()
        # Compares tensors in state dictionaries and checks that they are the same before/after saving
        generators_same = all(torch.equal(g_orig_state[key], g_saved_state[key]) for key in g_orig_state)

        d_orig_state = d_orig.state_dict()
        d_saved_state = d_saved.state_dict()
        # Compares tensors in state dictionaries and checks that they are the same before/after saving
        discriminators_same = all(torch.equal(d_orig_state[key], d_saved_state[key]) for key in d_orig_state)

        assert generators_same
        assert discriminators_same
        assert h_saved["epochs"] == 5
        assert h_saved["lr"] == 2e-5
        assert h_saved["dropout"] == 0.4
        assert h_saved["trbatch"] == 8
        assert h_saved["valbatch"] == 8

        