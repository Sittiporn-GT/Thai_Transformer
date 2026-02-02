# ------------------------------------------------------------------------------------------- #
# Data preparation utilities for plutonic rock thin-section image classification
import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

def prepare_data(
        data_dir: str = "plutonic_rocks",
        image_size : int = 512,
        batch_size : int = 32,
        num_workers : int = 4,
        train_sample_size: int | None = None,
        test_sample_size: int | None = None,
        val_sample_size: int | None = None,
        shuffle_train: bool = True,
        seed: int = 42,
):
    """
    Prepare PyTorch DataLoaders for the plutonic rock dataset.
    Expected directory structure:
        data_dir/
            train/<class_name>/image1.png
            test/<class_name>/image2.png
            val/<class_name>/image3.png
    Notes:
    - This uses the same preprocessing for train/val/test.
    - Normalization mean/std should be computed from the dataset for best results.
    """
# ------------------------------------------------------------------------------------------- #
# Image preprocessing and normalization
    mean = (0.50222724, 0.50163647, 0.49506611)
    std = (0.29474634, 0.29574405, 0.29541864)

    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")
    val_dir   = os.path.join(data_dir, "val")

    trainset = ImageFolder(root=train_dir, transform=base_transform)
    valset   = ImageFolder(root=val_dir, transform=base_transform)
    testset  = ImageFolder(root=test_dir, transform=base_transform)

    g = torch.Generator()
    g.manual_seed(seed)

    def maybe_subset(dataset, sample_size: int | None):
        if sample_size is None:
            return dataset
        n = len(dataset)
        if sample_size > n:
            raise ValueError(f"Sample size {sample_size} exceeds dataset size {n}.")
        indices = torch.randperm(n, generator=g)[:sample_size].tolist()
        return Subset(dataset, indices)
    
    # Apply subsetting 
    trainset = maybe_subset(trainset, train_sample_size)
    valset = maybe_subset(valset, val_sample_size)
    testset = maybe_subset(testset, test_sample_size)

    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    valloader   = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader  = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Extract classes 
    classes = trainset.dataset.classes if isinstance(trainset, Subset) else trainset.classes
    return trainloader, valloader, testloader, classes
# ------------------------------------------------------------------------------------------- #
