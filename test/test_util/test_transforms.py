import os
import pytest
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.utils.image_processing import Preprocessing

@pytest.mark.usefixtures("testImage")
class TestImageTransforms():
    def testCrop(self, testImage):
        """ Test crop transform 
        """
        cropsz = (96, 96)
        t = Preprocessing.createCropTransform(cropsz)
        ret = t(testImage)
        shp = ret.shape
        assert shp[0] == 3
        assert shp[1] == cropsz[0]
        assert shp[2] == cropsz[1]

        assert torch.max(ret) <= 1
        assert torch.min(ret) >= 0

    def testLRBicubic(self, testImage):
        cropsz = (96, 96)
        sf = 4
        lr_method = "bicubic"

        ct = Preprocessing.createCropTransform(cropsz)
        cropped = ct(testImage)
        lrt = Preprocessing.createLRTransform(lr_method, sf, None, None)
        lr = lrt(cropped)

        shp = lr.shape
        assert shp[0] == 3
        assert shp[1] == cropsz[0] // sf
        assert shp[2] == cropsz[1] // sf

        assert torch.max(lr) <= 1
        assert torch.min(lr) >= 0

    def testLRGaussian(self, testImage):
        cropsz = (96, 96)
        sf = 4
        lr_method = "gaussian"

        ct = Preprocessing.createCropTransform(cropsz)
        cropped = ct(testImage)
        lrt = Preprocessing.createLRTransform(lr_method, sf, (9,9), [1,1])
        lr = lrt(cropped)

        shp = lr.shape
        assert shp[0] == 3
        assert shp[1] == cropsz[0] // sf
        assert shp[2] == cropsz[1] // sf

        assert torch.max(lr) <= 1
        assert torch.min(lr) >= 0

    def testHR(self, testImage):
        cropsz = (96, 96)

        ct = Preprocessing.createCropTransform(cropsz)
        cropped = ct(testImage)
        hrt = Preprocessing.createHRTransform()
        hr = hrt(cropped)

        shp = hr.shape
        assert shp[0] == 3
        assert shp[1] == cropsz[0]
        assert shp[2] == cropsz[1]

        assert torch.max(hr) <= 1
        assert torch.min(hr) >= -1





