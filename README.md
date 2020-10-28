# FlowNet_1.0
Tensorflow 2.3 implementation of FlowNet 1.0 by Dosovitskiy et al.

This repository is tensorflow implementation of FlowNet 1.0, by Alexey Dosovitskiy et al. in tensorflow 2.3.

This code is designed to run on CPU only and does not support GPU.

The code can also handle the batchnorm variation of FlowNet 1.0.

The implementation of the code was derived from the Pytorch version of this code by [ClementPinard](https://github.com/ClementPinard/FlowNetPytorch). 

# Pretrained Weights

To download weights of the model, click here. The model relies on these checkpoints to work properly.

# Model Verification

When the model is run with the pretrained weights, it should predict the following result using the input images provided (image1 and image2 in the image folder).

![](images/result.png)

# Sources

[1] A. Dosovitskiy, P. Fischer, E. Ilg, P. Häusser, C. Hazırba¸s, V. Golkov, P. v.d. Smagt, D. Cremers, and T. Brox. Flownet: Learning optical flow with convolutional networks. In IEEE International Conference on Computer Vision (ICCV), 2015.

