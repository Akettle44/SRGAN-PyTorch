## Unit tests SRGAN

import torch
import os
import pytest

from src.train.train import PtTrainer
from torch.utils.data import DataLoader

class TestTraining():
    def test_TrainInit(self, generator, discriminator, data):
        """ Test training initialization
        """
        # TODO use fixtures to standardize generator/discriminator generation across tests
        trainer = PtTrainer(generator, discriminator, data)

        g_orig = generator.state_dict()
        g_trainer = trainer.generator.state_dict()
        d_orig = discriminator.state_dict()
        d_trainer = trainer.discriminator.state_dict()
        generators_same = all(torch.equal(g_orig[key], g_trainer[key]) for key in g_orig)
        discriminators_same = all(torch.equal(d_orig[key], d_trainer[key]) for key in d_orig)
        
        assert generators_same
        assert discriminators_same
        assert trainer.dataset.root_dir == os.path.join(os.getcwd(), "datasets/imagenet_test")
        assert trainer.dataset.blur_kernel_size == (5,9)
        assert trainer.dataset.sigma == (0.1,5.)
        assert trainer.dataset.batch_size == 32
        assert trainer.dataset.num_workers == 8
        assert len(trainer.dataset) == 1300
        assert len(trainer.dataset.train_dataset) == 910
        assert len(trainer.dataset.val_dataset) == 195
        assert len(trainer.dataset.test_dataset) == 195
        assert isinstance(trainer.dataset.train_loader, DataLoader)
        assert isinstance(trainer.dataset.val_loader, DataLoader)
        assert isinstance(trainer.dataset.test_loader, DataLoader)
    
    def test_DefaultOptimizer(self, generator, discriminator, data):
        trainer = PtTrainer(generator, discriminator, data)

        g_orig = list(trainer.generator.parameters())
        g_opt = trainer.g_optimizer.param_groups
        assert isinstance(trainer.g_optimizer, torch.optim.Adam)
        assert g_orig == g_opt[0]['params']
        assert g_opt[0]['lr'] == 2e-5

        d_orig = list(trainer.discriminator.parameters())
        d_opt = trainer.d_optimizer.param_groups
        assert isinstance(trainer.d_optimizer, torch.optim.Adam)
        assert d_orig == d_opt[0]['params']
        assert d_opt[0]['lr'] == 2e-5
    
    def test_SetHyps(self, generator, discriminator, data):
        trainer = PtTrainer(generator, discriminator, data)
        hyps = {
            "epochs":1,
            "lr":2e-5,
            "dropout":0.4,
            "freezeLayers":[6,7,8,9,10],
            "trbatch":5,
            "valbatch":10
        }
        trainer.setHyps(hyps)
        assert all(trainer.hyps[key] == hyps[key] for key in hyps.keys())

    def test_UpdateOptimizerLr(self, generator, discriminator, data):
        trainer = PtTrainer(generator, discriminator, data)
        hyps = {'lr': 999}
        trainer.setHyps(hyps)
        trainer.updateOptimizerLr()
        assert all(param_group['lr'] == 999 for param_group in trainer.g_optimizer.param_groups)

    def test_SetDevice(self, generator, discriminator, data):
        trainer = PtTrainer(generator, discriminator, data)
        trainer.setDevice("cpu")
        assert trainer.device == "cpu"
        



