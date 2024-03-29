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
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
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
    """
    # initialize input and output dimensions
    self.n_inputs = n_inputs
    self.n_classes = n_classes
    self.n_hidden = n_hidden

    # initialize the modules
    self.modules = []

    if not isinstance(n_hidden, list):
      n_hidden = [n_hidden]

    # Add the first linear layer (input -> hidden)
    self.modules.append(LinearModule(n_inputs, n_hidden[0], True))
    # Add the first Activation layer (hidden -> hidden)
    self.modules.append(ELUModule())

    # Add the rest of the linear layers (hidden -> hidden)
    for i in range(len(n_hidden) - 1):
      self.modules.append(LinearModule(n_hidden[i], n_hidden[i + 1], False))
      self.modules.append(ELUModule())

    # Add the last linear layer (hidden -> output)
    self.modules.append(LinearModule(n_hidden[-1], n_classes, False))
    # Add the last Activation layer (output -> output)
    self.modules.append(SoftMaxModule())

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through
    several layer transformations.

    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """

    # numpy implementation of the forward pass
    for module in self.modules:
      x = module.forward(x)
    
    return x


  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss.

    Args:
      dout: gradients of the loss

    """

    # numpy implementation of the backward pass
    for module in reversed(self.modules):
      dout = module.backward(dout)
    
    return dout

  def parameters(self):
    """
    Returns the parameters of the network.

    Returns:
      params: list of network parameters. Each element is a tuple (W, b)
    """

    # initialize the parameters
    params = []

    # add the parameters of each module
    for module in self.modules:
      params += module.parameters()
    
    return params
  
  def summary(self):
    """
    Prints the summary of the network.
    """

    print("----------------------------------------------------------------")
    print("Layer (type)            Output Shape        Param #     Trainable")
    print("================================================================")
    total_params = 0
    total_trainable_params = 0
    for module in self.modules:
      output_shape, params, trainable_params = module.summary()
      total_params += params
      total_trainable_params += trainable_params
      print("{:<23} {:<20} {:<12} {}".format(module.__class__.__name__, str(output_shape), str(params), str(trainable_params)))
    print("================================================================")
    print("Total params: " + str(total_params))
    print("Trainable params: " + str(total_trainable_params))
    print("Non-trainable params: " + str(total_params - total_trainable_params))
    print("----------------------------------------------------------------")
    
  def clear_cache(self):
    """
    Remove any saved tensors for the backward pass from any module.
    Used to clean-up model from any remaining input data when we want to save it.
    """

    # clear the cache of each module
    for module in self.modules:
      module.clear_cache()

