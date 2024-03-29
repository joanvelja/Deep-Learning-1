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
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
  

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

    device = torch.device("cpu") # running my code on the CPU

    num_classes = predictions.shape[1] 
    cm = np.zeros((num_classes, num_classes)) # confusion matrix is a 2D array of size [n_classes, n_classes]
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

    device = torch.device("cpu") # running my code on the CPU

    confusion_matrix = confusion_matrix.detach().numpy()

    # Compute the true positives, false positives and false negatives
    tp = np.diag(confusion_matrix)
    fp = np.sum(confusion_matrix, axis=0) - tp
    fn = np.sum(confusion_matrix, axis=1) - tp

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

    # Set default device
    device = torch.device("cpu") # running my code on the CPU

    # Set the model to evaluation mode
    model.eval() # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
                 # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/2
                 # torch.inference_mode() is used to disable gradient calculation and dropout/batch norm
                 # https://pytorch.org/docs/stable/generated/torch.inference_mode.html#torch.inference_mode
                 # allows to speed up the model evaluation

    
    # Initialize the confusion matrix
    cm = torch.zeros((num_classes, num_classes), device='cpu') # confusion matrix is a 2D array of size [n_classes, n_classes], using torch is faster than numpy
    # Initialize the loss
    loss_criterion = nn.CrossEntropyLoss()
    loss = 0

    # Loop over the dataset
    with torch.inference_mode(): # https://pytorch.org/docs/stable/generated/torch.inference_mode.html#torch.inference_mode
      for batch in data_loader:
          
          # Move tensors to MPS for faster computation
          inputs, targets = batch[0].to(device), batch[1].to(device)
          # Forward pass
          prediction = model(inputs)
          # Compute the loss
          loss += loss_criterion(prediction, targets)
          # Move tensors to CPU for NumPy operations
          prediction = prediction.cpu().detach().numpy()
          targets = targets.cpu().detach().numpy()
          # Compute the confusion matrix
          cm += confusion_matrix(prediction, targets)
  
    # Normalize the loss
    loss = loss.clone() / len(data_loader)
    # Compute the metrics
    cm = cm / cm.sum(dim=1, keepdim=True).float()
    metrics = confusion_matrix_to_metrics(cm)
    # Add the loss to the metrics
    metrics['loss'] = loss 
    metrics['confusion_matrix'] = cm

    return metrics


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
    
    if torch.backends.mps.is_available():
      torch.mps.manual_seed(seed)

        
    # Set default device
    device = torch.device("cpu")

    # Loading the dataset
    train, val = cifar10_utils.get_cifar10(data_dir)
    
    # Initialize model and loss module
    model = MLP(hidden_dims, use_batch_norm).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Training loop including validation
    val_accuracies = []
    best_model = None
    best_val_accuracy = 0
    logging_info = {}
    losses = []
    epoch_losses = []

    # Training loop
    for epoch in tqdm(range(epochs)):
      epoch_loss = 0

      # Set the model to training mode
      model.train()
      for batch in cifar10_loader['train']:

        # Reset the gradients
        optimizer.zero_grad()

        # Move tensors to device
        inputs, labels = batch[0].to(device), batch[1].to(device)

        # Forward pass
        prediction = model(inputs)

        # Compute the loss
        loss = criterion(prediction, labels)
        epoch_loss += loss

        # Backward pass
        loss.backward()
        optimizer.step()
        loss_np = loss.clone().detach().numpy()
        losses.append(loss_np)

      # Compute the validation accuracy
      val_accuracy = evaluate_model(model, cifar10_loader['validation'])['accuracy']
      val_accuracies.append(val_accuracy)

      # Save the best model
      if val_accuracy > best_val_accuracy:
        best_model = deepcopy(model)
        best_val_accuracy = val_accuracy
      

      # Save the loss
      logging_info['loss'] = epoch_loss / len(cifar10_loader['train'])
      epoch_losses.append(epoch_loss / len(cifar10_loader['train']))

      # Save the validation accuracy
      logging_info['val_accuracy'] = val_accuracy
      logging_info['train_loss'] = losses
      logging_info['epoch_losses'] = epoch_losses

      print(f"Epoch {epoch+1} loss: {logging_info['loss']}, validation accuracy: {logging_info['val_accuracy']}")

    # Evaluate the best model on the test set
    test_metrics = evaluate_model(best_model, cifar10_loader['test'])

    print(f"\nFinal results:")
    print("===============")
    print(f"Best validation accuracy: {best_val_accuracy}")
    print(f"Test accuracy: {test_metrics['accuracy']}")

    print(f"precision: {test_metrics['precision']}")
    print(f"recall: {test_metrics['recall']}")
    print(f"f1_beta: {test_metrics['f1_beta']}\n")

    return model, best_model, val_accuracies, test_metrics, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_false',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
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

    model, best_model, val_accuracies, test_accuracy, logging_info = train(**kwargs)

    # Plot the loss curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(logging_info['train_loss'], color='darkblue', linestyle='-', linewidth=2, label='Iteration Loss')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Loss Curve", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the loss curve per epoch
    epoch_losses_cpu = [tensor.cpu().detach().numpy() for tensor in logging_info['epoch_losses']]
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_losses_cpu, color='green', linestyle='-', marker='s', markersize=5, label='Epoch Loss')
    for i, value in enumerate(epoch_losses_cpu):
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

    # Plot the confusion matrix
    import itertools
    cifar10 = cifar10_utils.get_cifar10('data/')
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=128,
                                                  return_numpy=False)
    cm = evaluate_model(best_model, cifar10_loader['test'])['confusion_matrix']

    cm_np = cm.numpy()
    cm_np = np.round(cm_np, decimals=3)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_np, interpolation='nearest', cmap='Blues') 
    plt.title("Confusion Matrix for CIFAR-10 Classification", fontsize=15)
    plt.colorbar()

    classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    plt.tight_layout()
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Annotate each cell with the numeric value
    thresh = cm_np.max() / 2. # threshold for the text color to be white
    for i, j in itertools.product(range(cm_np.shape[0]), range(cm_np.shape[1])):
      plt.text(j, i, format(cm_np[i, j], '.3f'),
          horizontalalignment="center",
          color="white" if cm_np[i, j] > thresh else "black",
          fontsize=8)

    plt.show()

    # Print f1_beta scores for different values of beta
    print('\n' + '=' * 50 + '\n')
    for beta in [0.1, 1, 10]:
      metrics = confusion_matrix_to_metrics(cm, beta=beta)
      print(f"f1_{beta}: {metrics['f1_beta']}")
      print('#' * 50 + '\n')