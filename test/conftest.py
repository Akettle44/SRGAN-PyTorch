### Shared functions for unit tests ###

import os
import pytest
from src.model.model import Generator, Discriminator
from src.data.imagenet import ImageNetDataset

@pytest.fixture(scope='class')
def setUp():
    root_dir = os.getcwd()
    test_dir = os.path.join(root_dir, "test")
    model_dir = os.path.join(root_dir, "models")
    return test_dir, root_dir, model_dir

@pytest.fixture
def tearDown():
    pass

@pytest.fixture
def generator():
    return Generator(scale=1)

@pytest.fixture
def discriminator():
    return Discriminator()

@pytest.fixture
def data():
    root_dir = os.path.join(os.getcwd(), "datasets/imagenet_test")
    blur_kernel_size = (5,9)
    sigma = (0.1,5.)
    batch_size = 32
    num_workers = 8
    imagenet = ImageNetDataset(root_dir, blur_kernel_size, sigma, batch_size, num_workers)
    imagenet.createDataloaders(imagenet.batch_size, imagenet.num_workers, [0.7,0.15,0.15])
    return imagenet
