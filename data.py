import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

def prepare_data(batch_size=64, num_workers=4, train_sample_size=None, test_sample_size=None, val_sample_size=None, shuffle_train=True, seed=None):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.50222724, 0.50163647, 0.49506611),
            std=(0.29474634, 0.29574405, 0.29541864)
        )])

    train_set = ImageFolder(root="/dataset/train", transform=transform)
    val_set   = ImageFolder(root="/dataset/val",   transform=transform)
    test_set  = ImageFolder(root="/dataset/test",  transform=transform)

    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    def make_subset(ds, n):
        if n is None:
            return ds
        n = min(n, len(ds))
        idx = torch.randperm(len(ds), generator=g)[:n]
        return Subset(ds, idx)

    train_set = make_subset(train_set, train_sample_size)
    val_set   = make_subset(val_set,   val_sample_size)
    test_set  = make_subset(test_set,  test_sample_size)
    
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    valloader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    classes = train_set.dataset.classes if isinstance(train_set, Subset) else train_set.classes

    return trainloader, valloader, testloader, classes
