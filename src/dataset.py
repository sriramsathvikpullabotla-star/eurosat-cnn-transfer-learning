import torch
from torchvision.datasets import EuroSAT
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset

# =============================================================================
# 1. DYNAMIC SUBSET WRAPPER
# =============================================================================
class TransformSubset(Dataset):
    """
    A lightweight wrapper around a PyTorch Subset.
    
    Problem: When torch.utils.data.random_split splits a dataset, the resulting
    Subsets point to the parent dataset and share the exact same transform pipeline.
    
    Solution: This wrapper intercepts __getitem__, pulls the raw image from the 
    underlying Subset, and applies a dataset-specific transform pipeline dynamically.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        # 1. Fetch raw PIL image and ground-truth integer class label (0-9)
        image, label = self.subset[index]
        
        # 2. Apply augmentation or normalization transform on-the-fly in RAM
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def __len__(self):
        # Returns the total number of samples allocated to this specific split
        return len(self.subset)


# =============================================================================
# 2. DATALOADER FACTORY FUNCTION
# =============================================================================
def get_dataloaders(data_dir='./data', batch_size=32, use_augmentation=False):
    """
    Downloads EuroSAT (if not cached), performs a deterministic 70/15/15 split,
    and returns production-ready PyTorch DataLoaders for train, val, and test.
    
    Args:
        data_dir (str): Local folder to store/read the downloaded EuroSAT archive.
        batch_size (int): Number of images processed simultaneously in parallel.
        use_augmentation (bool): If True, applies random flips and rotations to train split.
        
    Returns:
        train_loader, val_loader, test_loader: Iterable PyTorch DataLoader objects.
    """
    # -------------------------------------------------------------------------
    # A. REPRODUCIBLE SPLITTING (70% Train, 15% Val, 15% Test)
    # -------------------------------------------------------------------------
    # Fixed seed generator guarantees the exact same image indices end up in
    # train, val, and test across every run, experiment, and model architecture.
    generator = torch.Generator().manual_seed(42)

    # Download raw EuroSAT RGB dataset (27,000 images, 64x64 pixels, 10 classes)
    raw_dataset = EuroSAT(root=data_dir, download=True)
    
    total = len(raw_dataset)               # Exactly 27,000 images
    train_len = int(total * 0.70)          # 18,900 images
    val_len = int(total * 0.15)            # 4,050 images
    # Subtract to absorb any integer truncation or float rounding differences
    test_len = total - train_len - val_len  # 4,050 images
    
    # Split raw dataset deterministically into three disjoint subsets
    train_sub, val_sub, test_sub = random_split(
        raw_dataset, [train_len, val_len, test_len], generator=generator
    )

    # -------------------------------------------------------------------------
    # B. TRANSFORM PIPELINES
    # -------------------------------------------------------------------------
    # Base Transform: Applied to Validation and Testing (and Train when ablation is off)
    # - ToTensor(): Converts PIL image (0-255, HxWxC) to float32 tensor (0.0-1.0, CxHxW).
    # - Normalize(): Standardizes channels using ImageNet mean & std values. This centers
    #   features around 0 with unit variance, stabilizing gradient propagation and matching
    #   the input distribution expected by pre-trained ResNet models.
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Augmentation Transform: Applied ONLY to Training data when use_augmentation=True
    # - RandomHorizontalFlip(): Flips images horizontally with 50% probability.
    # - RandomRotation(15): Rotates images randomly between -15 and +15 degrees.
    # Satellite images are orientation-invariant (crops/fields look valid from any direction),
    # forcing the model to learn invariant spatial features rather than memorizing exact pixels.
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Select pipeline for training based on experiment flag
    train_transform = aug_transform if use_augmentation else base_transform

    # -------------------------------------------------------------------------
    # C. BIND TRANSFORMS TO SPLITS
    # -------------------------------------------------------------------------
    train_dataset = TransformSubset(train_sub, transform=train_transform)
    # Validation and test sets NEVER receive data augmentation
    val_dataset = TransformSubset(val_sub, transform=base_transform)
    test_dataset = TransformSubset(test_sub, transform=base_transform)

    # -------------------------------------------------------------------------
    # D. BUILD ITERABLE DATALOADERS
    # -------------------------------------------------------------------------
    # - shuffle=True (Train): Shuffles batch order every epoch to prevent pattern memorization.
    # - shuffle=False (Val/Test): Preserves order for consistent, deterministic evaluation.
    # - num_workers=2: Spawns 2 CPU background worker processes to prefetch and augment batches
    #   while the GPU is executing forward/backward passes, preventing hardware starvation.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
