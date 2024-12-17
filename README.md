# Pytorch-SRGAN

## Table of contents
- [Pytorch-SRGAN](#Pytorch-SRGAN)
  - [Overview](#Overview)
  - [Requirements](#Requirements)
  - [Dataset](#Dataset)
  - [Preprocessing](#Preprocessing)
  - [Training](#Training)
  - [Configurations](#Configurations)
  - [Results](#Results)
  - [Contributors](#Contributors)

## Overview
This repository re-implements [SRGAN](https://arxiv.org/abs/1609.04802v5) in a modular setting that allows for eaiser experimentation. SRGAN is a conditional GAN that is capable of doing super resolution of input images. The generator is fully convolutional, allowing it to work with any image size. We trained on numerous datasets and experimented with various architectures for the generator and discriminator, as well as different loss functions. 
The details of these experiments can be found in the writeup under docs. 

## Requirements
Use the following command to install required packages.
```
pip install -r /path/to/requirements.txt
```
We included a main file that is a useful starting point. We run this file from the root directory using `python -m main` using Python 3.10.12. In the future, we intend to extend main to include argparse support. We structured the code as a python module if you would rather call its functions in a Jupyter notebook.

## Dataset(s)
We trained on two datasets, [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [ImageNet](https://image-net.org/download.php). We used the CIFAR-10 images in their native resolution (32x32x3) and took 96x96x3 crops of the imagenet images during training.

## Preprocessing
Super-resolution uses a blurring kernel to create low resolution (LR) images from the high resolution (HR) ground-truths. In our experiments, we preprocessed using gaussian blur and bicubic interpolation operations. The blurring kernel size was proportional to the image size and the downsampling factor was the scale (user input). For higher resolution images (e.g. imagenet), we sampled a 96x96x3 crop as the LR image. The low resolution images were scaled between [-1, 1] and the high resolution labels [0, 1], following the SRGAN paper. 

## Training
We trained our GAN using a two step approach. In the first step, pretraining, we train the generator alone using MSE loss. The idea of this step is to produce a stable starting point before introducing stochasticity related to typical conditional GAN training. After completing pretraining, we train the generator and discriminator jointly. The generator can be configured to use either MSE or a feature based loss in conjunction with the typical adversarial term. The feature loss computes the MSE of feature maps sampled from VGG. The discriminator can be trained using classic cross-entropy loss or a variant described in [lsgan](https://arxiv.org/pdf/1506.05751) that aims to provide stabler gradients. 

## Configurations
We used yaml files for the dataset and model configuration to make our repository as flexible as possible. These configs are stored in the "configs" directory at the root level. These files allow the user the modify the model architecture as well as training hyperparameters. A copy of this file is saved after each training session in case large changes are being made. 

## Results
CIFAR10:
We found super-resolution difficult to work with on CIFAR10. We trained hundreds of models, but were never able to ...

ImageNet: TBD

## Contributors
Andrew Kettle (Lead), Feilong Hou, Steven Chang