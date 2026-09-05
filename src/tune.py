import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders
from src.model import SimpleCNN
from src.train import train_one_epoch
from src.evaluate import evaluate_model

def objective(trial):
    """
    Optuna objective function.
    Defines search space, trains a model, and returns validation accuracy.
    """
    # -------------------------------------------------------------------------
    # 1. HYPERPARAMETER SEARCH SPACE
    # -------------------------------------------------------------------------
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    dropout_rate = trial.suggest_float("dropout", 0.2, 0.5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -------------------------------------------------------------------------
    # 2. DATA LOADERS
    # -------------------------------------------------------------------------
    # Using batch_size=32 and standard base transforms for tuning
    train_loader, val_loader, _ = get_dataloaders(batch_size=32, use_augmentation=False)
    
    # -------------------------------------------------------------------------
    # 3. INSTANTIATE MODEL & OPTIMIZER
    # -------------------------------------------------------------------------
    model = SimpleCNN(num_classes=10, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
    # -------------------------------------------------------------------------
    # 4. MINI TRAINING LOOP (3 Epochs per trial for rapid search)
    # -------------------------------------------------------------------------
    num_epochs = 3
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
        # Report intermediate score to enable Optuna's pruning algorithm
        trial.report(val_acc, epoch)
        
        # Abort unpromising trials early to save compute time
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return best_val_acc

def run_optimization(n_trials=10):
    """
    Initializes the study and runs optimization trials.
    """
    print("Starting Optuna Study...")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
    )
    study.optimize(objective, n_trials=n_trials)
    
    print("\n" + "=" * 40)
    print("OPTIMIZATION FINISHED")
    print("=" * 40)
    print(f"Best Validation Accuracy: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, val in study.best_params.items():
        print(f"  {key}: {val}")
        
    return study
