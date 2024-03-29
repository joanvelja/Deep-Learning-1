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

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

from cifar100_utils import get_train_validation_set, get_test_set, set_dataset

# Initialize weights of the model with given mean and std
def init_weights(module, mean, std):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=mean, std=std)
        module.bias.data.zero_()
        return module
    else:
        raise NameError('Cannot initialize weights for non linear modules.')

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Randomly initialize and modify the model's last layer for CIFAR100.
    print(model.parameters)
    for param in model.parameters():
      param.requires_grad = False

    old_fc = model.fc
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= num_classes, bias=True)
    new_fc = init_weights(new_fc, mean = 0, std = 0.01)
    model.fc = new_fc

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None, augmentation_params=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """

    model.to(device)

    # Load the datasets
    train, val = get_train_validation_set(data_dir, augmentation_name=augmentation_name, augmentation_params=augmentation_params)
    collate_fn = None
    train_dataloader      = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers = 8, drop_last=True,
                                       collate_fn=collate_fn)
    validation_dataloader = DataLoader(dataset=val, batch_size=batch_size, shuffle=False, num_workers = 8, drop_last=False,
                                       collate_fn=collate_fn)
    

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()

    val_accuracies= []
    best_val_accuracy = 0

    # Training loop with validation after each epoch. Save the best model.
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        model.train()
        for batch in tqdm(train_dataloader, leave=False, desc=f"Epoch {epoch+1}"):
            
            # Zero gradients
            optimizer.zero_grad()

            # Move tensors to device
            inputs, labels = batch[0].to(device), batch[1].to(device)

            # Forward pass
            prediction = model(inputs)

            # Compute the loss
            loss = criterion(prediction, labels)
            epoch_loss += loss

            # Backpropagate
            loss.backward()
            optimizer.step()
            
        # Evaluate on validation  
        val_accuracy = evaluate_model(model, validation_dataloader, device)
        val_accuracies.append(val_accuracy)

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_model = deepcopy(model)
            best_val_accuracy = val_accuracy
            # Checkpoint the best model
            torch.save(best_model.state_dict(), checkpoint_name)

        epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} loss: {epoch_loss}, validation accuracy: {val_accuracy}")

    return best_model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    
    # Set the model to evaluation mode
    model.eval() # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
                 # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/2
                 # torch.inference_mode() is used to disable gradient calculation and dropout/batch norm
                 # https://pytorch.org/docs/stable/generated/torch.inference_mode.html#torch.inference_mode
                 # allows to speed up the model evaluation

    
    # loss_criterion = nn.CrossEntropyLoss() # To be used if we want to compute the loss on the test/validation set
    accuracy = 0.0

    # Loop over the dataset
    with torch.inference_mode(): # https://pytorch.org/docs/stable/generated/torch.inference_mode.html#torch.inference_mode
      for batch in data_loader:
          
          # Move tensors to device for faster computation
          inputs, targets = batch[0].to(device), batch[1].to(device)
          # Forward pass
          prediction = model(inputs)
          # Accuracy
          pred_labels = torch.argmax(prediction, dim=1)
          accuracy += (pred_labels == targets).float().mean().item()
        
    # Normalize the accuracy
    accuracy /= len(data_loader)
    
    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load the model
    model = get_model().to(device)

    # Get the augmentation to use
    

    # Train the model
    best_model = train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name = 'best_ResNet', device=device, augmentation_name=augmentation_name)

    # Evaluate the model on the test set
    test = get_test_set(data_dir, test_noise)
    test_dataloader =  DataLoader(dataset=test, batch_size=batch_size, shuffle=False, drop_last=False,
                                       collate_fn=None)
    accuracy = evaluate_model(best_model, test_dataloader, device)
    print(f"The accuracy of the test set is {accuracy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar100', 'cifar10'],
                        help='Dataset to use.')
    parser.add_argument('--augmentation_name', default="AutoAugment", type=str,
                        help='Augmentation to use.')
    parser.add_argument('--test_noise', default=False, action="store_true",
                        help='Whether to test the model on noisy images or not.')
    parser.add_argument('--augmentation_params', default={'p': 0.2}, type=dict,
                        help='Augmentation parameters to use.')

    args = parser.parse_args()
    kwargs = vars(args)
    set_dataset(kwargs.pop('dataset'))
    main(**kwargs)
