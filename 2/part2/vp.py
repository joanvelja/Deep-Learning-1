################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

"""Defines various kinds of visual-prompting modules for images."""
import torch
import torch.nn as nn
import numpy as np


class FixedPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a fixed patch over an image.
    For refernece, this prompt should look like Fig 2(a) in the PDF.
    """
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

        # Define the prompt parameters here. The prompt is basically a
        # patch (can define as self.patch) of size [prompt_size, prompt_size]
        # that is placed at the top-left corner of the image.

        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn
        patch = torch.randn(1, 3, args.prompt_size, args.prompt_size)
        self.patch = nn.Parameter(patch)


    def forward(self, x):
        # TODO: For a given batch of images, place the patch at the top-left

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        prompt = self.patch
        batch_size = x.size(0)
        # Expanding patch to match the batch size and adding it to the image
        prompt = self.patch.expand(batch_size, -1, -1, -1)
        clone = x.clone()
        clone[:, :, :self.patch.shape[2], :self.patch.shape[3]] += prompt
        

        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        #viz = x[0].permute(1, 2, 0).detach().cpu().numpy()
        #viz = np.clip(viz, 0, 1)
        #import matplotlib.pyplot as plt
        #plt.imshow(viz)
        #plt.show()

        return clone


class PadPrompter(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size
        self.pad_size = pad_size # This is going to be used in the forward function for the pad_left and pad_right
        

        # TODO: Define the padding as variables self.pad_left, self.pad_right, self.pad_up, self.pad_down

        # Hints:
        # - Each of these are parameters that we need to learn. So how would you define them in torch?
        # - See Fig 2(c) in the assignment to get a sense of how each of these should look like.
        # - Shape of self.pad_up and self.pad_down should be (1, 3, pad_size, image_size)
        # - See Fig 2.(g)/(h) and think about the shape of self.pad_left and self.pad_right
        #self.pad_up = nn.Parameter(torch.randn(1, 3, pad_size, image_size)) # 1, 3, 30, 224
        #self.pad_down = nn.Parameter(torch.randn(1, 3, pad_size, image_size)) # 1, 3, 30, 224

        # Top right and left corners are covered by self.pad_up; Bottom right and left corners are covered by self.pad_down
        # Thus, we need to adjust the height of the left and right pads since after adding top and bottom pads the image size changes!
        #adjusted_height = image_size + 2 * pad_size
        #self.pad_left = nn.Parameter(torch.randn(1, 3, adjusted_height, pad_size)) # 1, 3, 284, 30
        #self.pad_right = nn.Parameter(torch.randn(1, 3, adjusted_height, pad_size)) # 1, 3, 284, 30


        self.pad_up = nn.Parameter(torch.randn(1, 3, pad_size, image_size))
        self.pad_down = nn.Parameter(torch.randn(1, 3, pad_size, image_size))
        
        
        # Pad left and pad right should account for the top and bottom corners being already padded by pad_up and pad_down
        # Their height should be image_size - 2 * pad_size

        self.pad_left = nn.Parameter(torch.randn(1, 3, image_size - 2 * pad_size, pad_size))
        self.pad_right = nn.Parameter(torch.randn(1, 3, image_size - 2 * pad_size, pad_size))
        
        #print(f"pad_up shape: {self.pad_up.shape}")
        #print(f"pad_down shape: {self.pad_down.shape}")
        #print(f"pad_left shape: {self.pad_left.shape}")
        #print(f"pad_right shape: {self.pad_right.shape}")

    def forward(self, x):
        # TODO: For a given batch of images, add the prompt as a padding to the image.

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.
        batch_size = x.size(0)
        pad_up = self.pad_up.expand(batch_size, -1, -1, -1)
        pad_down = self.pad_down.expand(batch_size, -1, -1, -1)
        pad_left = self.pad_left.expand(batch_size, -1, -1, -1)
        pad_right = self.pad_right.expand(batch_size, -1, -1, -1)

        # Concatenating pads to the image
        # The torch.cat function in PyTorch is used to concatenate tensors along a specified dimension. When concatenating tensors, 
        # all dimensions except the one you're concatenating along must match.
        # For example, if you concatenate along dimension 0 (let's say this is the batch dimension in a batch of images), 
        # the other dimensions (height, width, channels) of the tensors being concatenated must be the same.
        # In this case, we are concatenating along dimension 2 (height), so the other dimensions (batch, width, channels) must be the same.
        # The pads are concatenated to the image along the height dimension, so the width and channels dimensions must be the same.
        
        #x = torch.cat([pad_up, x, pad_down], dim=2)
        x[:, :, :self.pad_up.size(2), :] = pad_up
        x[:, :, -self.pad_down.size(2):, :] = pad_down
        x[:, :, self.pad_size:x.shape[2]-self.pad_size, :self.pad_left.size(3)] = pad_left
        x[:, :, self.pad_size:x.shape[2]-self.pad_size, -self.pad_right.size(3):] = pad_right
    
        #print(f"image shape after adding pads to the image: {x.shape}")

        #viz = x[0].permute(1, 2, 0).detach().cpu().numpy()
        #viz = np.clip(viz, 0, 1)
        #import matplotlib.pyplot as plt
        #plt.imshow(viz)
        #plt.show()

        return x