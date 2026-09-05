import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs one full pass (epoch) over the training dataset.
    
    Args:
        model: The neural network instance (SimpleCNN or ResNet).
        dataloader: PyTorch DataLoader yielding (images, labels) batches.
        criterion: Loss function (nn.CrossEntropyLoss).
        optimizer: Optimization algorithm updating weights (e.g., Adam, SGD).
        device: Hardware device to run on (torch.device('cuda') or torch.device('cpu')).
        
    Returns:
        epoch_loss (float): Mean CrossEntropy loss across the entire training split.
        epoch_acc (float): Top-1 classification accuracy (ratio of correct / total, 0.0 to 1.0).
    """
    # -------------------------------------------------------------------------
    # 1. SWITCH MODEL TO TRAINING MODE
    # -------------------------------------------------------------------------
    # Activates layers that behave differently during training vs inference:
    # - Dropout: Randomly deactivates neurons with probability p to reduce overfitting.
    # - BatchNorm: Computes and updates running mean and variance per batch.
    model.train()
    
    # Accumulator variables to compute true weighted averages across the full epoch
    running_loss = 0.0  # Sum of raw loss values for every processed image
    correct = 0       # Total count of correctly predicted labels
    total = 0         # Total number of images processed so far
    
    # -------------------------------------------------------------------------
    # 2. BATCH ITERATION LOOP
    # -------------------------------------------------------------------------
    # tqdm provides an animated terminal progress bar for monitoring speed (it/s)
    # leave=False cleans up the progress bar once the epoch finishes to avoid notebook clutter
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        
        # Move batch tensors from CPU host RAM to accelerator memory (GPU VRAM or CPU)
        # inputs shape: (batch_size, 3, 64, 64)
        # labels shape: (batch_size,) containing class integer indices 0 through 9
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # ---------------------------------------------------------------------
        # 3. GRADIENT BUFFER RESET
        # ---------------------------------------------------------------------
        # PyTorch accumulates (sums) gradients into param.grad by default.
        # If not explicitly zeroed, gradients from the previous batch will add to
        # current gradients, leading to incorrect updates and unstable divergence.
        optimizer.zero_grad()
        
        # ---------------------------------------------------------------------
        # 4. FORWARD PASS
        # ---------------------------------------------------------------------
        # Feed inputs through convolutional and linear layers.
        # outputs shape: (batch_size, 10) representing unnormalized log-odds (logits)
        outputs = model(inputs)
        
        # ---------------------------------------------------------------------
        # 5. COMPUTE LOSS
        # ---------------------------------------------------------------------
        # nn.CrossEntropyLoss internally applies LogSoftmax + NLLLoss (Negative Log Likelihood).
        # It measures discrepancy between predicted class distribution and true label index.
        # By default, loss is the MEAN value across the current mini-batch (scalar float tensor).
        loss = criterion(outputs, labels)
        
        # ---------------------------------------------------------------------
        # 6. BACKWARD PASS (AUTOGRAD CHAIN RULE)
        # ---------------------------------------------------------------------
        # Differentiates loss w.r.t. all model parameters that have requires_grad=True.
        # Traverses computation graph in reverse; stores partial derivatives in param.grad.
        loss.backward()
        
        # ---------------------------------------------------------------------
        # 7. PARAMETER UPDATE (OPTIMIZER STEP)
        # ---------------------------------------------------------------------
        # Updates weights using calculated gradients according to optimizer rule:
        # e.g., for SGD: W = W - (learning_rate * W.grad)
        # e.g., for Adam: updates moving first and second moments before stepping.
        optimizer.step()
        
        # ---------------------------------------------------------------------
        # 8. METRIC ACCUMULATION
        # ---------------------------------------------------------------------
        # loss.item() extracts the scalar value from the 0-dim PyTorch tensor.
        # Because loss is an average over the current batch, multiplying by inputs.size(0)
        # recovers the unnormalized sum of losses. This prevents skew if the final batch
        # in the dataset has fewer items than batch_size.
        running_loss += loss.item() * inputs.size(0)
        
        # outputs.max(dim=1) searches across columns (classes) for each row (image).
        # Returns a named tuple: (values, indices).
        # We discard max values (_) and keep the class indices with highest scores (predicted).
        # predicted shape: (batch_size,)
        _, predicted = outputs.max(1)
        
        # Increment sample count by the number of elements in the current batch
        total += labels.size(0)
        
        # predicted.eq(labels) produces a boolean tensor: True if matched, False otherwise.
        # .sum() counts the True occurrences in the batch tensor.
        # .item() converts the 1-element tensor into a native Python integer.
        correct += predicted.eq(labels).sum().item()
        
    # -------------------------------------------------------------------------
    # 9. EPOCH-LEVEL METRIC COMPUTATION
    # -------------------------------------------------------------------------
    # Divide total accumulated loss and correct predictions by overall sample count
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc
