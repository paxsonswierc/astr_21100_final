import torch
import numpy as np
from itertools import cycle
torch.manual_seed(222222)
np.random.seed(222222)

def dann_training_loop(source_train_dataloader, source_val_dataloader, target_train_dataloader, target_val_dataloader, 
                       model, optimizer, task_loss_fn, domain_classifier_loss_fn, source_y_domain, target_y_domain, batch_size):
    '''
    Inputs:
        source_train_dataloader = pytorch dataloader with sdss training images and labels, for training
        source_val_dataloader = pytorch dataloader with sdss training images and labels, for validation
        target_train_dataloader = pytorch dataloader with hsc training images and labels, for training
        target_val_dataloader = pytorch dataloader with hsc training images and labels, for validation 
        model = DANN pytorch model
        optimizer = Pytorch optimizer
        task_loss_fn = Pytorch loss function for the task - cross entropy
        domain_classifier_loss_fn = Pytorch loss function for the domain classifier - binary cross entropy
        source_y_domain = Pytorch tensor of labels for source domain, with size (batch size), zeros recommended
        target_y_domain = Pytorch tensor of labels for target domain, with size (batch size), ones recommended
        batch_size = size of batches
    Returns: List of floats, each a stat cooresponding to :
        0 - source_estimate_training_accuracy 
        1 - source_estimate_training_loss 
        2 - source_domain_training_loss 
        3 - target_estimate_training_accuracy 
        4 - target_estimate_training_loss 
        5 - target_domain_training_loss
        6 - source_estimate_val_accuracy 
        7 - source_estimate_val_loss 
        8 - source_domain_val_loss
        9 - target_estimate_val_accuracy 
        10 - target_estimate_val_loss
        11 - target_domain_val_loss
        12 - dc_accuracy
    '''
    # Training loop first
    # Initiate source and target stats for the epoch
    source_estimate_training_correct = 0.
    source_estimate_training_loss = 0.
    source_domain_correct = 0.
    source_domain_training_loss = 0.

    target_estimate_training_correct = 0.
    target_estimate_training_loss = 0.
    target_domain_correct = 0.
    target_domain_training_loss = 0.
    # Since the target dataloader is shorter, we cycle it until getting through the source dataloader
    for source_data, target_data in zip(source_train_dataloader, cycle(target_train_dataloader)): # Training loop
         # Source training
        X_source, y_source = source_data
        X_source, y_source = X_source.cuda(), y_source.reshape(-1).cuda() # Move data to gpu and make sure labels are the right shape
        model.zero_grad() # Reset gradient for all model parameters
        source_estimate, source_domain = model(X_source) # Get model estimates and guess at domains
        source_estimate_loss = task_loss_fn(source_estimate, y_source) # Get the loss of the task estimate of source data
        source_domain_loss = domain_classifier_loss_fn(source_domain, source_y_domain) # Get the loss of the domain guess
        # Update stats for source data
        _, source_predicted = torch.max(source_estimate.data, 1)
        source_estimate_training_correct += torch.sum((source_predicted == y_source)).item()
        source_estimate_training_loss += source_estimate_loss.item()
        source_domain_correct += len(source_domain[source_domain < .5])
        source_domain_training_loss += source_domain_loss.item()
        # Target training (same steps as source)
        X_target, y_target = target_data
        X_target, y_target = X_target.cuda(), y_target.reshape(-1).cuda()
        target_estimate, target_domain = model(X_target) # Remember our target estimate here is only for tracking model performance
        # The target estimate loss is ONLY for tracking performance, not used to update parameters
        # In an actual DA scenario, we wouldn't have access to the target labels
        target_estimate_loss = task_loss_fn(target_estimate, y_target)
        target_domain_loss = domain_classifier_loss_fn(target_domain, target_y_domain) # This WILL be used to update parameters
        # Update stats for target data
        _, target_predicted = torch.max(target_estimate.data, 1)
        target_estimate_training_correct += torch.sum((target_predicted == y_target)).item()
        target_estimate_training_loss += target_estimate_loss.item()
        target_domain_correct += len(target_domain[target_domain > .5])
        target_domain_training_loss += target_domain_loss.item()
        # Add the losses used to update parameters
        loss = source_estimate_loss + source_domain_loss + target_domain_loss
        loss.backward() # Get the gradient
        optimizer.step() # Update the network

    len_train_dataset = len(source_train_dataloader) # Used for stats
    # Update stats for the whole epoch during training
    source_estimate_training_accuracy = source_estimate_training_correct / (len_train_dataset * batch_size)
    source_estimate_training_loss /= len_train_dataset
    source_domain_training_loss /= len_train_dataset

    target_estimate_training_accuracy = target_estimate_training_correct / (len_train_dataset * batch_size)
    target_estimate_training_loss /= len_train_dataset
    target_domain_training_loss /= len_train_dataset

    dc_accuracy = (source_domain_correct + target_domain_correct) / ( 2 * (len_train_dataset * batch_size))
    # Validation loop for source data
    source_estimate_val_correct = 0.
    source_estimate_val_total = 0.
    source_estimate_val_loss = 0.  
    source_domain_val_loss = 0.    
    for data in source_val_dataloader:
        with torch.no_grad():
            X, y = data
            X, y = X.cuda(), y.reshape(-1).cuda()
            estimate, domain = model(X)
            estimate_loss = task_loss_fn(estimate, y)
            domain_loss = domain_classifier_loss_fn(domain, source_y_domain)
            _, predicted = torch.max(estimate.data, 1)
            source_estimate_val_correct += torch.sum((predicted == y)).item()
            source_estimate_val_total += y.size(0)
            source_estimate_val_loss += estimate_loss.item()
            source_domain_val_loss += domain_loss.item()
    
    source_estimate_val_accuracy = source_estimate_val_correct / source_estimate_val_total
    source_estimate_val_loss /= len(source_val_dataloader)
    source_domain_val_loss /= len(source_val_dataloader)
    # Validation loop for target data
    target_estimate_val_correct = 0.
    target_estimate_val_total = 0.
    target_estimate_val_loss = 0.  
    target_domain_val_loss = 0.    
    for data in target_val_dataloader:
        with torch.no_grad():
            X, y = data
            X, y = X.cuda(), y.reshape(-1).cuda()
            estimate, domain = model(X)
            estimate_loss = task_loss_fn(estimate, y)
            domain_loss = domain_classifier_loss_fn(domain, target_y_domain)
            _, predicted = torch.max(estimate.data, 1)
            target_estimate_val_correct += torch.sum((predicted == y)).item()
            target_estimate_val_total += y.size(0)
            target_estimate_val_loss += estimate_loss.item()
            target_domain_val_loss += domain_loss.item()
    
    target_estimate_val_accuracy = target_estimate_val_correct / target_estimate_val_total
    target_estimate_val_loss /= len(target_val_dataloader)
    target_domain_val_loss /= len(target_val_dataloader)
    # Collect all the stats into a list
    stats = [source_estimate_training_accuracy, source_estimate_training_loss, source_domain_training_loss, 
                 target_estimate_training_accuracy, target_estimate_training_loss, target_domain_training_loss,
                 source_estimate_val_accuracy, source_estimate_val_loss, source_domain_val_loss,
                 target_estimate_val_accuracy, target_estimate_val_loss, target_domain_val_loss, dc_accuracy]

    return stats

def dann_test_loop(dataloader, model, loss_fn):
    '''
    This tests the task part of the DANN network
    Input:
        dataloader - test dataloader for either domain
        model - trained DANN model
        loss_fn - Loss function for the task network (cross entropy)
    Returns:
        Accuracy - float of accuracy of task predictions
        Loss - float of loss of the task predictions
        Predictions and true values for test set (both are lists of the tensors)
    '''
    correct = 0.
    total = 0.
    all_loss = 0.   
    predictions = []
    true = []  
    for data in dataloader:
        with torch.no_grad():
            X, y = data
            X, y = X.cuda(), y.reshape(-1).cuda()
            preds, _ = model(X) # Since we are in testing, we are only looking at task predictions
            loss = loss_fn(preds, y)
            _, predicted = torch.max(preds.data, 1)
            correct += torch.sum((predicted == y)).item()
            total += y.size(0)
            all_loss += loss.item()
            predictions.append(predicted.cpu())
            true.append(y.cpu())
            
    accuracy = correct / total
    all_loss = all_loss / len(dataloader)
    
    return accuracy, all_loss, predictions, true