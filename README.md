## Table of contents
- [Pytorch-SRGAN](#Pytorch-SRGAN)
  - [Overview](#Overview)
  - [Requirements](#Requirements)
  - [Data Set](#Data-Set)
  - [Results](#Results)


# Pytorch-SRGAN
## Overview
This project aims to implement SRGAN with pytorch base on paper 
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802v5)

## Requirement
  Use
  ```
  pip install -r /path/to/requirements.txt
  ```
  to install required packages

## Data Set
  We had 2 datasets. We used [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) as an initial experiment to verify the model is working. However due to the limited size of images in CIFAR10 (32x32), the model was struggling to learn a good super-resolution sample distribution. \
  Thus we moved on to [ImageNet](https://image-net.org/download.php). Due to limited computational power, we first trained on 3,900 images containing only lions, tigers, and cheetahs. After achieving significant results, we move on to a bigger subset of ImageNet containing 23,869 images of different dog breeds.

## Results
Upscale Factor = 4
The left is down-sampled image, the middle is super resolution image(output of the SRGAN), and the right is high-resolution image.
![Generated_images](https://github.com/Akettle44/SRGAN-PyTorch/blob/Update-readme/figures/full_feat.jpg?raw=true)
And the training loss curve is:
![loss](https://github.com/Akettle44/SRGAN-PyTorch/blob/Update-readme/figures/feat_loss_epoch(6)_graph.png?raw=true)
