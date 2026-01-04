import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

def prepare_data(batch_size=4, num_workers=2, train_sample_size=None, val_sample_size=None, test_sample_size=None, shuffle_train=True):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1024, 1024)),
        transforms.Normalize((0.50222724, 0.50163647, 0.49506611), (0.29474634, 0.29574405, 0.29541864)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1024, 1024)),
        transforms.Normalize((0.50222724, 0.50163647, 0.49506611), (0.29474634, 0.29574405, 0.29541864)),
    ])

    # Load datasets
    trainset = ImageFolder(root='plutonic_rocks_split_02/train', transform=train_transform)
    valset = ImageFolder(root='plutonic_rocks_split_02/val', transform=test_transform)
    testset = ImageFolder(root='plutonic_rocks_split_02/test', transform=test_transform)

    # Subset if necessary
    if train_sample_size is not None:
        trainset = Subset(trainset, torch.randperm(len(trainset))[:train_sample_size])
    
    if val_sample_size is not None:
        valset = Subset(valset, torch.randperm(len(valset))[:val_sample_size])
    
    if test_sample_size is not None:
        testset = Subset(testset, torch.randperm(len(testset))[:test_sample_size])

    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Extract class names safely
    classes = trainset.dataset.classes if isinstance(trainset, Subset) else trainset.classes

    return trainloader, valloader, testloader, classes
