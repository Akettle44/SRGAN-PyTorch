### Shared functions for unit tests ###

import os
import pytest

@pytest.fixture(scope='class')
def setUp():
    root_dir = os.getcwd()
    test_dir = os.path.join(root_dir, "test")
    model_dir = os.path.join(root_dir, "models")
    return test_dir, root_dir, model_dir

@pytest.fixture
def tearDown():
    pass
