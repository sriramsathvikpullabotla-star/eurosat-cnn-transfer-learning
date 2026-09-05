import torch
from tqdm import tqdm

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on a validation or test dataset.
    
    Args:
        model: Trained neural network instance (SimpleCNN or ResNet).
        dataloader: DataLoader yielding evaluation batches (val_loader or test_loader).
        criterion: Loss function (nn.CrossEntropyLoss) to track loss on unseen data.
        device: Hardware device (torch.device('cuda') or torch.device('cpu')).
        
    Returns:
        eval_loss (float): Average CrossEntropy loss across the evaluation split.
        eval_acc (float): Accuracy ratio (correct / total, 0.0 to 1.0).
    """
    # -------------------------------------------------------------------------
    # 1. SWITCH MODEL TO INFERENCE MODE
    # -------------------------------------------------------------------------
    # - Dropout: Completely disabled (p=0.0) so all neurons contribute.
    # - BatchNorm: Freezes statistics; uses running mean/var learned during training.
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # -------------------------------------------------------------------------
    # 2. DISABLE GRADIENT CALCULATION
    # -------------------------------------------------------------------------
    # torch.no_grad() deactivates the Autograd engine.
    # Memory footprint drops sharply because intermediate activation history is discarded.
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move data to the same target hardware device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass only (no backward pass or optimizer step)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Multiply average batch loss by batch size to sum raw loss
            running_loss += loss.item() * inputs.size(0)
            
            # Find class with highest logit score
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    # Compute final metrics across all validation/test samples
    eval_loss = running_loss / total
    eval_acc = correct / total
    
    return eval_loss, eval_acc
