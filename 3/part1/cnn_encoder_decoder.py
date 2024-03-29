################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch.nn as nn
import torch


class CNNEncoder(nn.Module):
    def __init__(self, num_input_channels: int = 1, num_filters: int = 32,
                 z_dim: int = 20):
        """Encoder with a CNN network
        Inputs:
            num_input_channels - Number of input channels of the image. For
                                 MNIST, this parameter is 1
            num_filters - Number of channels we use in the first convolutional
                          layers. Deeper layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()
        self.to('cuda' if torch.cuda.is_available() else 'mps')

        # For an intial architecture, you can use the encoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.

        # The encoder should output two vectors of shape [B,z_dim], one for the mean and one for the log_std.
        # Note that the encoder does not need to have a softmax or log_softmax layer at the end, as we will use a loss that already applies a softmax to the output of the encoder.

        # Encoder
        # Tutorial 9 code
        c_hid = num_filters
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            nn.GELU(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            nn.GELU(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*c_hid, 128),
            nn.GELU(),
            nn.Linear(128, 2* z_dim)
        )

    def forward(self, x):
        """
        Inputs:
            x - Input batch with images of shape [B,C,H,W] of type long with values between 0 and 15.
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """
        x = x.float() / 15 * 2.0 - 1.0  # Move images between -1 and 1

        return self.net(x).chunk(2, dim=-1) # Chunk tuples the result into (mean, log_std)
    
    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device


class CNNDecoder(nn.Module):
    def __init__(self, num_input_channels: int = 16, num_filters: int = 32,
                 z_dim: int = 20):
        """Decoder with a CNN network.
        Inputs:
            num_input_channels - Number of channels of the image to
                                 reconstruct. For a 4-bit MNIST, this parameter is 16
            num_filters - Number of filters we use in the last convolutional
                          layers. Early layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()
        self.to('cuda' if torch.cuda.is_available() else 'mps')
        

        # For an intial architecture, you can use the decoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        c_hid = num_filters
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2*16*c_hid),
            nn.GELU(),
            nn.Unflatten(1, (2*c_hid, 4, 4)), # To avoid flattening explicitly during forward pass
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2, output_padding=1), # 4x4 => 8x8
            nn.GELU(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=3, stride=2), # 16x16 => 28x28
        )

    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:
            x - Prediction of the reconstructed image based on z.
                This should be a logit output *without* a softmax applied on it.
                Shape: [B,num_input_channels,28,28]
        """
        return self.net(z.to(self.device))

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device
