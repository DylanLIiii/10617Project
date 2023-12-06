from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn 
import torch.nn.functional as F
import os 
import torch.optim as optim
import numpy as np
# if you want to use wandb, uncomment the following lines
import wandb 
from tqdm import tqdm
from termcolor import colored
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
import torch.optim.lr_scheduler as lr_scheduler


class EarlyStopping():
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_checkpoint = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint = True
            print(f"INFO: Early stopping counter reset to {self.counter}, best loss updated to {self.best_loss}")
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            self.save_checkpoint = False
            if self.counter >= np.round(self.patience * 0.5):
                print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
            

def make_dataset(X, test_size=0.2):
    """
    Create train and test datasets from input data.

    Args:
        X (numpy.ndarray): Input data array.
        test_size (float, optional): Proportion of the data to include in the test dataset. Defaults to 0.2.

    Returns:
        tuple: A tuple containing the train dataset and test dataset.
    """
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
    train_dataset = TensorDataset(torch.from_numpy(X_train).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float())
    return train_dataset, test_dataset


def make_dataloader(train, test, batch_size=32):
    """Create data loaders for training and testing.

    Args:
        train (Dataset): The training dataset.
        test (Dataset): The testing dataset.
        batch_size (int, optional): The batch size for the data loaders. Defaults to 32.

    Returns:
        tuple: A tuple containing the training data loader and testing data loader.
    """
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train(args, model, optim, criterion, train_loader, val_loader, epochs, device, beta_list, wandb_logging=False) -> tuple:
    """
    Trains a model using the specified optimizer, criterion, and data loaders for a given number of epochs.

    Args:
        model (torch.nn.Module): The model to be trained.
        optim (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function used for training.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        epochs (int): The number of epochs to train the model.
        device (torch.device): The device to be used for training.
        wandb_logging (bool, optional): Whether to log training progress using wandb. Defaults to False.

    Returns:
        tuple: A tuple containing the trained model, a list of training losses, and a list of validation losses.
    """
    model = model.to(device)
    
    # Initialize empty lists to store training and validation losses
    train_losses = []
    train_recon_losses = []
    train_kl_losses = []
    val_losses = []
    val_recon_losses = []
    val_kl_losses = []
    
    # Enable logging with wandb if wandb_logging is True
    if wandb_logging:
        wandb.watch(model, criterion, log="all", log_freq=20)

    #scheduler = lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.1, patience=args.lr_patience, verbose=True)
    earlystopping = EarlyStopping(patience=args.stop_patience, min_delta=1e-5)
    # Loop over the specified number of epochs
    for epoch in tqdm(range(epochs), desc=colored("Training", "green"), unit="epoch"): 
        train_loss = 0.0
        val_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        recon_val_loss = 0.0
        kl_val_loss = 0.0
        
        
        # Set the model to training mode
        model.train()
        
        # Iterate over the training data loader
        for batch_idx, (data) in enumerate(train_loader):
            data = data[0].to(device)
            
            # Zero the gradients
            optim.zero_grad()
            
            # Forward pass
            if args.vae == True:
                output, latent, mu, logvar  = model(data)
                recon_loss, kl_loss = vae_loss(output, data, mu, logvar)
                loss = recon_loss + beta_list[epoch] * kl_loss
                recon_loss += recon_loss
                kl_loss += kl_loss
            else:
                output, latent = model(data)
                # Compute the loss
                loss = criterion(output, data)
        
            # Backward pass
            loss.backward()
            
            # Update the weights
            optim.step()
            
            # Accumulate the training loss
            train_loss += loss.item()

        
        # Append the average training loss for the epoch to the train_losses list
        train_losses.append(train_loss / len(train_loader))
        
        if args.vae: 
            train_recon_losses.append(recon_loss/ len(train_loader))
            train_kl_losses.append(kl_loss / len(train_loader))
        
        
        # Set the model to evaluation mode
        model.eval()
        
        # Disable gradient calculation
        with torch.no_grad():
            # Iterate over the validation data loader
            for (data) in val_loader:
                data = data[0].to(device)
                
                # Forward pass
                if args.vae == True:
                    output, latent, mu, logvar  = model(data)
                    recon_loss, kl_loss = vae_loss(output, data, mu, logvar)
                    loss = recon_loss + beta_list[epoch] * kl_loss
                    recon_val_loss += recon_loss
                    kl_val_loss += kl_loss
                else:
                    output, latent = model(data)
                    # Compute the loss
                    loss = criterion(output, data)
                    
                
                # Accumulate the validatio  n loss
                val_loss += loss.item()

        
        # Append the average validation loss for the epoch to the val_losses list
        val_losses.append(val_loss / len(val_loader))
        #scheduler.step(val_losses[-1])
            
        
        if args.vae:
            val_recon_losses.append(recon_val_loss/ len(val_loader))
            val_kl_losses.append(kl_val_loss / len(val_loader))
            earlystopping(val_recon_losses[-1])
        else:
            earlystopping(val_losses[-1])
        
        if earlystopping.early_stop:
            print("Early stopped training at epoch %d" % epoch)
            break # terminate the training loop
        
        # save best model 
        if earlystopping.save_checkpoint:
            torch.save(model.state_dict(), os.path.join(args.model_dir, f"{model.__class__.__name__}_best.pth"))
            print("INFO: Saved best model at epoch %d" % epoch)

        current_lr = optim.param_groups[0]["lr"]
        # report loss evert 10 epochs 
        if epoch % 20 == 0:
            print(colored(f"Epoch {epoch} | Training loss: {train_losses[-1]}", 'green'))
            print(colored(f"Epoch {epoch} | Validation loss: {val_losses[-1]}", 'blue'))
            
            print(colored(f"Epoch {epoch} | Current learning rate: {current_lr}", 'blue'))
            if args.vae: 
                print(colored(f"Epoch {epoch} | Train Reconstruction loss: {train_recon_losses[-1]}", 'green'))   
                print(colored(f"Epoch {epoch} | Train KL loss: {train_kl_losses[-1]}", 'green'))
                print(colored(f"Epoch {epoch} | Val Reconstruction loss: {val_recon_losses[-1]}", 'blue'))
                print(colored(f"Epoch {epoch} | Val KL loss: {val_kl_losses[-1]}", 'blue'))
            
        # Log training and validation losses to wandb if wandb_logging is True
        if wandb_logging:
            wandb.log({"Training Loss": train_losses[-1], "Validation Loss": val_losses[-1], "epoch": epoch})
            wandb.log({"Learning Rate": current_lr})
            if args.vae:
                wandb.log({"Train Reconstruction Loss": train_recon_losses[-1], "Train KL Loss": train_kl_losses[-1]})
                wandb.log({"Val Reconstruction Loss": val_recon_losses[-1], "Val KL Loss": val_kl_losses[-1]})
    
    # Return the best trained model, training losses, and validation losses
    best_model = model.load_state_dict(torch.load(os.path.join(args.model_dir, f"{model.__class__.__name__}_best.pth")))
    return model, train_losses, val_losses

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x)
    KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD


