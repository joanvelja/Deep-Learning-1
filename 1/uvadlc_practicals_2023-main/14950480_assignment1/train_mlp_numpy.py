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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """
    num_classes = predictions.shape[1] 
    cm = np.zeros((num_classes, num_classes)) 
    pred_labels = np.argmax(predictions, axis=1) 
    for t, p in zip(targets, pred_labels): 
        cm[t, p] += 1

    return cm


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
  """
  Converts a confusion matrix to accuracy, precision, recall and f1 scores.
  Args:
    confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
  Returns: a dictionary with the following keys:
    accuracy: scalar float, the accuracy of the confusion matrix
    precision: 1D float array of size [n_classes], the precision for each class
    recall: 1D float array of size [n_classes], the recall for each clas
    f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
  """
  
  # Compute the true positives, false positives, false negatives and true negatives
  tp = np.diag(confusion_matrix)
  fp = np.sum(confusion_matrix, axis=0) - tp
  fn = np.sum(confusion_matrix, axis=1) - tp
  tn = np.sum(confusion_matrix) - (tp + fp + fn)

  # Compute the metrics
  accuracy = np.sum(tp) / np.sum(confusion_matrix)
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  f1_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


  return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_beta': f1_beta}


def evaluate_model(model, data_loader, num_classes=10):
  """
  Performs the evaluation of the MLP model on a given dataset.

  Args:
    model: An instance of 'MLP', the model to evaluate.
    data_loader: The data loader of the dataset to evaluate.
  Returns:
    metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.
  """

  # Initialize the confusion matrix
  cm = np.zeros((num_classes, num_classes))
  # Initialize the loss
  loss = 0

  # Loop over the dataset
  for batch in data_loader:

    # Forward pass
    forward_pass = model.forward(batch[0])
    # Compute the loss
    loss += CrossEntropyModule().forward(forward_pass, batch[1])
    # Compute the confusion matrix
    cm += confusion_matrix(forward_pass, batch[1])

  # Normalize the loss
  loss /= len(data_loader)
  # Compute the metrics
  metrics = confusion_matrix_to_metrics(cm)
  # Add the loss to the metrics
  metrics['loss'] = loss

  return metrics



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    # Initialize model and loss module
    model = MLP(3 * 32 * 32, hidden_dims, 10)
    criterion = CrossEntropyModule()

    # Training loop including validation
    val_accuracies = []
    best_model = None
    best_val_accuracy = 0
    logging_dict = {}
    losses = []
    epoch_losses = []

    # Training loop
    for epoch in tqdm(range(epochs)):
      epoch_loss = 0

      # Train the model for one epoch
      for batch in cifar10_loader['train']:

        # Clear cache
        model.clear_cache() # simulates the zero_grad() method of PyTorch
        # Clear gradients
        for module in model.modules: # simulates the zero_grad() method of PyTorch
            if hasattr(module, 'grads'):
                module.grads = {name: np.zeros_like(param) for name, param in module.params.items()}

        # Forward pass
        forward_pass = model.forward(batch[0])
        # Compute loss
        loss = criterion.forward(forward_pass, batch[1])
        epoch_loss += loss
        # Backward pass
        grad = criterion.backward(forward_pass, batch[1])
        model.backward(grad)
        # Update parameters
        for module in model.modules:
          if hasattr(module, 'params'):
            module.params['weight'] -= lr * module.grads['weight']
            module.params['bias'] -= lr * module.grads['bias']
        
        losses.append(loss)
      
      # Normalize the loss
      epoch_loss /= len(cifar10_loader['train'])

      # Log the loss
      epoch_losses.append(epoch_loss)

      # Evaluate the model on the whole validation set
      val_accuracy = evaluate_model(model, cifar10_loader['validation'])['accuracy']
      val_accuracies.append(val_accuracy)

      print(f"Epoch {epoch+1} loss: {epoch_loss}, validation accuracy: {val_accuracy}")
  
      # Save the best model, i.e. the one that performs best on the validation set
      if val_accuracy > best_val_accuracy:
          best_model = deepcopy(model)
          best_val_accuracy = val_accuracy
    
    # Evaluate the best model on the test set
    test_metrics = evaluate_model(best_model, cifar10_loader['test'])
    test_accuracy = test_metrics['accuracy']

    print(f"\nFinal results:")
    print("===============")
    print(f"Best validation accuracy: {best_val_accuracy}")
    print(f"Test accuracy: {test_metrics['accuracy']}")

    print(f"precision: {test_metrics['precision']}")
    print(f"recall: {test_metrics['recall']}")
    print(f"f1_beta: {test_metrics['f1_beta']}\n")

    # Save logging information
    logging_dict['best_model'] = best_model
    logging_dict['val_accuracies'] = val_accuracies
    logging_dict['test_accuracy'] = test_accuracy
    logging_dict['losses'] = losses
    logging_dict['epoch_losses'] = epoch_losses

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    best_model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    
    # Plot the loss curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(logging_dict['losses'], color='darkblue', linestyle='-', linewidth=2, label='Iteration Loss')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Loss Curve", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the loss curve per epoch
    plt.figure(figsize=(8, 6))
    plt.plot(logging_dict['epoch_losses'], color='green', linestyle='-', marker='s', markersize=5, label='Epoch Loss')
    for i, value in enumerate(logging_dict['epoch_losses']):
      plt.text(i, value, f"{value:.2f}", ha='center', va='bottom')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Loss Curve per Epoch", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the validation accuracies
    plt.figure(figsize=(8, 6))
    plt.plot(val_accuracies, color='red', linestyle='-', marker='^', markersize=5, label='Validation Accuracy')
    for i, value in enumerate(val_accuracies):
      plt.text(i, value, f"{value:.2f}", ha='center', va='bottom')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Validation Accuracies", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
