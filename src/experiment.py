import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from src.dataset import get_dataloaders
from src.model import SimpleCNN, get_resnet_transfer
from src.train import train_one_epoch
from src.evaluate import evaluate_model

def run_experiment(stage_name, model, train_loader, val_loader, test_loader, optimizer, criterion, device, epochs=10):
    """
    Executes a complete training and evaluation protocol for an experimental stage.
    
    Workflow:
      1. Trains the model across N epochs using train_loader.
      2. Validates performance after every epoch using val_loader.
      3. Saves the model weights (state_dict) whenever validation accuracy hits a new high.
      4. Restores the top-performing checkpoint after training finishes.
      5. Runs a final evaluation on the untouched test_loader to report an unbiased test metric.
      
    Args:
        stage_name (str): Unique label for logging and checkpoint naming (e.g. 'stage1_baseline').
        model: PyTorch model instance (SimpleCNN or ResNet).
        train_loader: DataLoader yielding training batches.
        val_loader: DataLoader yielding validation batches for model selection.
        test_loader: DataLoader reserved strictly for final testing (no hyperparameter feedback).
        optimizer: Configured PyTorch optimizer (e.g., SGD, Adam).
        criterion: Loss criterion (nn.CrossEntropyLoss).
        device: Target computation device (torch.device('cuda') or torch.device('cpu')).
        epochs (int): Number of complete passes over the training set (default: 10).
        
    Returns:
        test_acc (float): Accuracy achieved on the unseen test set by the best checkpoint.
        history_df (pd.DataFrame): Per-epoch record of loss and accuracy for visualization.
    """
    # -------------------------------------------------------------------------
    # 1. CHECKPOINT DIRECTORY SETUP
    # -------------------------------------------------------------------------
    # Create local storage folder for serialization; exist_ok=True avoids errors if it already exists
    os.makedirs("checkpoints", exist_ok=True)
    
    # Track the highest validation accuracy observed so far (used for early checkpointing)
    best_val_acc = 0.0
    checkpoint_path = f"checkpoints/{stage_name}_best.pth"
    
    # List of per-epoch metric dictionaries, later transformed into a pandas DataFrame
    history = []
    
    print(f"\n{'='*50}\nRUNNING: {stage_name}\n{'='*50}")
    
    # -------------------------------------------------------------------------
    # 2. TRAINING AND VALIDATION LOOP
    # -------------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        # Pass 1: Run one complete training epoch over the training set (weights get updated)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Pass 2: Evaluate model on the validation split (weights frozen, no gradients)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        # Record epoch metrics to analyze loss convergence and detect potential overfitting
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        # Display epoch progress with formatted loss and percentage accuracy
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}%")
        
        # ---------------------------------------------------------------------
        # 3. BEST MODEL CHECKPOINTING
        # ---------------------------------------------------------------------
        # If the current epoch beats all previous validation accuracy scores:
        # Save ONLY the state dictionary (learned tensor weights/biases), not the whole object.
        # This protects against overfitting in later epochs where validation accuracy might drop.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            
    # -------------------------------------------------------------------------
    # 4. FINAL UNBIASED TEST EVALUATION
    # -------------------------------------------------------------------------
    # Load the best recorded weights from disk back into the model architecture
    # (replaces the weights from epoch 10 with whichever epoch performed best on validation)
    model.load_state_dict(torch.load(checkpoint_path))
    
    # Evaluate strictly once on the held-out test set
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    
    print(f"\n--> {stage_name} Complete | Final Test Accuracy: {test_acc*100:.2f}%\n")
    
    # Return final test accuracy and DataFrame containing the training trajectory
    return test_acc, pd.DataFrame(history)
