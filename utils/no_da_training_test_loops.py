import torch
import numpy as np
torch.manual_seed(222222)
np.random.seed(222222)

def training_loop(train_dataloader, val_dataloader, model, optimizer, loss_fn):
    '''
    Inputs:
        train_dataloader - Pytorch dataloader 
        val_dataloader - Pytorch dataloader 
        model - CNN pytorch model
        optimizer - Optimizer to use
        loss_fn - Loss function for task
    Returns: 
        training_accuracy, training_loss, val_accuracy, val_loss (all floats)
    '''
    # Training
    training_correct = 0.
    training_total = 0.
    training_loss = 0.
    for data in train_dataloader:
        X, y = data
        X, y = X.cuda(), y.reshape(-1).cuda() # Put data on gpu and make sure labels are right shape
        optimizer.zero_grad() # Reset gradient of model parameters
        preds = model(X) # Get model outputs
        loss = loss_fn(preds, y) 
        loss.backward() # Compute gradient of model parameters
        optimizer.step() # Updat model
        # Get stats
        _, predicted = torch.max(preds.data, 1)
        training_correct += torch.sum((predicted == y)).item()
        training_total += y.size(0)
        training_loss += loss.item()
        
    training_accuracy = training_correct / training_total
    training_loss = training_loss / len(train_dataloader)
    # Validation
    val_correct = 0.
    val_total = 0.
    val_loss = 0.    
    for data in val_dataloader:
        with torch.no_grad():
            X, y = data
            X, y = X.cuda(), y.reshape(-1).cuda()
            preds = model(X)
            loss = loss_fn(preds, y)
            _, predicted = torch.max(preds.data, 1)
            val_correct += torch.sum((predicted == y)).item()
            val_total += y.size(0)
            val_loss += loss.item()
            
    val_accuracy = val_correct / val_total
    val_loss = val_loss / len(val_dataloader)
    
    return training_accuracy, training_loss, val_accuracy, val_loss

def test_loop(dataloader, model, loss_fn):
    '''
    Inputs:
        dataloader - pytorch dataloader
        model - CNN model that has been trained
        loss_fn - Loss function to evaluate model
    Returns:
        Accuracy and loss of the testing run (both floats)
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
            preds = model(X)
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