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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        Steps:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """        

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        if input_layer:
            # First layer doesn't have nonlinearity applied to
            self.params['weight'] = np.random.randn(in_features, out_features) * np.sqrt(1 / in_features)
        else:
            # Initialize weights using Kaiming initialization
            self.params['weight'] = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)

        # Initialize biases with zeros
        self.params['bias'] = np.zeros(out_features)

        # Initialize gradients with zeros
        self.grads['weight'] = np.zeros((in_features, out_features))
        self.grads['bias'] = np.zeros(out_features)


    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """

        # Flatten the input tensor 
        x_flat = x.reshape(x.shape[0], -1)

        # Implement forward pass of the module.
        out = (x_flat @ self.params['weight']) + self.params['bias']
        
        # Store intermediate variables inside the object. They can be used in backward pass computation.
        self.x = x_flat
        self.out = out
        
        return out
  

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """

        # Implement backward pass of the module.
        dx = dout @ self.params['weight'].T
        self.grads['weight'] = self.x.T @ dout
        self.grads['bias'] = np.sum(dout, axis=0)

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        
        # Set any cache to None.
        self.x = None
        self.out = None

    def summary(self):
        """
        Prints the summary of the layer.
        """

        # Print the summary of the layer
        output_shape = self.params['weight'].shape
        params = np.prod(output_shape) + output_shape[1]
        trainable_params = np.prod(output_shape) + output_shape[1]

        return output_shape, params, trainable_params


class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module
        """
        
        # Implement forward pass of the module.
        out = np.where(x > 0, x, np.exp(x) - 1)
        
        # Store intermediate variables inside the object. They can be used in backward pass computation.
        self.x = x
        self.out = out

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """

        # Implement backward pass of the module.
        dx = np.where(self.x > 0, dout, dout * np.exp(self.x))

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """
        # Set any cache to None.
        self.x = None
        self.out = None

    def summary(self):
        """
        Prints the summary of the layer.
        """

        # This module has no parameters
        output_shape = None
        params = 0
        trainable_params = 0

        return output_shape, params, trainable_params
        
class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
        """

        # Implement forward pass of the module, making use of the Max Trick
        x_max = np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x - x_max)
        out = x_exp / np.sum(x_exp, axis=1, keepdims=True)

        # Store intermediate variables inside the object. They can be used in backward pass computation.
        self.x = x
        self.out = out
        
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
        """

        # I tried to make the backward pass as efficient as possible, by using np.einsum instead of a for loop.
        # The implementation with the for loop is commented out here.
        #batch_size, num_classes = self.out.shape
        #dx = np.zeros_like(self.x)

        #for i in range(batch_size):
        #    y = self.out[i]
        #    jacobian_matrix = np.diagflat(y) - np.outer(y, y) # np.diagflat(y) is a diagonal matrix with the elements of y on the diagonal, where y is a vector.
        #                                                      # np.outer(y, y) is the outer product of y with itself, which is a matrix with the elements of y on the diagonal.
        #    dx[i] = np.dot(jacobian_matrix, dout[i])


        # Compute the Jacobian matrix of the softmax function
        # Note: The Jacobian matrix of the softmax function is a diagonal matrix
        y = self.out
        jacobian = y[:, :, np.newaxis] * (np.eye(y.shape[1]) - y[:, np.newaxis, :]) # nothing but sigma * (I - sigma), which in math notation is sigma * (1 - sigma)
                                                                                    # we add a new axis to y to make it a 3D tensor, so that we can perform a dot product with dout
                                                                                    # where each slice of the 3D tensor is multiplied with the corresponding slice of dout
                                                                                    # and summed over.
        # Using np.einsum to compute the gradient of the loss with respect to the inputs
        # Convoluted, but faster than using a for loop.
        # The gradient of the loss with respect to the inputs is computed as follows:
        dx = np.einsum('ij,ijk->ik', dout, jacobian) # einsum('ij,ijk->ik', dout, jacobian) is equivalent to np.dot(dout, jacobian)
                                                     # but faster and more memory efficient. Source: https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
                                                     # What einsum does is to perform a dot product between the first axis of dout and the second axis of jacobian.
                                                     # --> 'ij,ijk->ik' means indeed that the first axis of dout is multiplied with the first axis of jacobian and summed over.
                                                     # The result is a matrix with the same shape as the input x.
                                                     # The first axis of dout is the batch size, the second axis is the number of classes.
                                                     # The second axis of jacobian is the number of classes, the third axis is the number of classes.
                                                     

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.
        """

        # Set any cache to None.
        self.x = None
        self.out = None

    def summary(self):
        """
        Prints the summary of the layer.
        """

        # This module has no parameters
        output_shape = None
        params = 0
        trainable_params = 0

        return output_shape, params, trainable_params

class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
        """

        # Compute the softmax of the input
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # Compute the cross-entropy loss
        batch_size = x.shape[0]
        y = np.eye(x.shape[1])[y] # one-hot encoding
        out = -np.sum(y * np.log(softmax + 1e-9)) / batch_size


        # A faster implementation of the cross-entropy loss is the following (not using the one hot encoding):
        #log_probs = -np.log(softmax[np.arange(batch_size), y] + 1e-9)
        #out = np.sum(log_probs) / batch_size

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
        """

        ############################### NOTE ###############################################
        # I mimicked the implementation of PyTorch's CrossEntropyLoss                      #
        # Source: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html #
        ####################################################################################

        
        # Compute the softmax of the input
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # Compute the gradient of the cross-entropy loss with respect to the input
        batch_size = x.shape[0]
        y = np.eye(x.shape[1])[y] # one-hot encoding
        dx = (softmax - y) / batch_size

        # A faster implementation of the gradient of the cross-entropy loss with respect to the input is the following (not using the one hot encoding):
        #softmax[np.arange(batch_size), y] -= 1 # subtract 1 from the correct class (the class that is equal to 1 in the one-hot encoding)
        #dx = softmax / batch_size

        return dx