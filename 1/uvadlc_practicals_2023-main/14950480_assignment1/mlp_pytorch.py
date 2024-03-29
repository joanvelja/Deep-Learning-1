################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import OrderedDict


# I implemented a new MLP class to leverage the acceleration of PyTorch: https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html
# This allows to move the model to the MPS accelarator in my case, through super().__init__()


class MLP(nn.Module):
  '''
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
  '''
  def __init__(self, hidden_dims, use_batch_norm=False):
    '''
   Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.
    '''
    super().__init__()
    self.use_batch_norm = use_batch_norm
    self.hidden_dims = hidden_dims

    self.layers = nn.Sequential()
    for i in range(len(hidden_dims)):
            if i == 0:
                layer = nn.Linear(32 * 32 * 3, hidden_dims[i])
            else:
                layer = nn.Linear(hidden_dims[i-1], hidden_dims[i])

            # Kaiming Initialization
            nn.init.kaiming_normal_(layer.weight)  # 'relu' can be used for ELU as well

            self.layers.add_module('layer_{}'.format(i), layer)

            if use_batch_norm:
                self.layers.add_module('batch_norm_{}'.format(i), nn.BatchNorm1d(hidden_dims[i]))
            self.layers.add_module('activation_{}'.format(i), nn.ELU())

    self.layers.add_module('output_layer', nn.Linear(hidden_dims[-1], 10))

    # No softmax here because it is included in the loss function nn.CrossEntropyLoss()
    # https://discuss.pytorch.org/t/is-softmax-mandatory-in-a-neural-network/149862

  def forward(self, x):
    '''
    Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
    '''
    # Flatten the input tensor
    x = x.view(x.shape[0], -1) # flatten the input tensor to (batch_size, 3 * 32 * 32)
    
    # Apply the layers
    output = self.layers(x)

    return output
  
  @property
  def device(self):
      """
      Returns the device on which the model is. Can be useful in some situations.
      """
      return next(self.parameters()).device