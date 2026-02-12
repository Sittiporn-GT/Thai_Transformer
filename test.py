import os
import json
import torch
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import ThaiTForClassification

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(42)

# -------------------- Config & Patchs -------------------- #
exp_dir = "/content/drive/MyDrive/experiments/ViT/ViT-B/16"
checkpoint_path = os.path.join(exp_dir, "model_best.pt")
config_path = os.path.join(exp_dir, "config.json")
out_dir = os.path.join(exp_dir, "eval_outputs")
os.makedirs(out_dir, exist_ok=True)

with open(config_path, "r") as f:
    config = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- Load model -------------------- #
model = ThaiTForClassification(config).to(device)

ckpt = torch.load(checkpoint_path, map_location="cpu")
state = ckpt
if isinstance(ckpt, dict):
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
        
missing, unexpected = model.load_state_dict(state, strict=False)
print("Loaded checkpoint.")
print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")
if len(missing) > 0:
    print("  Example missing keys:", missing[:10])
if len(unexpected) > 0:
    print("  Example unexpected keys:", unexpected[:10])

model.eval()

# -------------------- DataLoader -------------------- #
use_prepare = False
try:
    from data import prepare_data
    use_prepare = True
except ImportError:
    use_prepare = False

if use_prepare:
    trainloader, _, testloader, class_names = prepare_data(batch_size=64)
else:
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    train_dir = "/dataset/train"
    test_dir  = "/dataset/test"
    transform = transforms.Compose([
        transforms.Resize((config.get("image_size", 512), config.get("image_size", 512))),
        transforms.ToTensor(),
        transforms.Normalize(
            tuple(config.get("mean", (0.44591194, 0.4460215, 0.44402866))),
            tuple(config.get("std",  (0.27868111, 0.28104728, 0.2775477)))
        )
    ])
    train_ds = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    test_ds  = torchvision.datasets.ImageFolder(root=test_dir,  transform=transform)
    
    trainloader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    testloader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    class_names = train_ds.classes

# -------------------- Evaluation Function -------------------- #
def evaluate_and_plot(dataloader, split_name="Train"):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            logits, _ = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(f"\n=== {split_name} Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(1.0 + 0.5*len(class_names), 1.0 + 0.5*len(class_names)))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues")
    plt.title(f"{split_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
# -------------------- Run ------------------- #
evaluate_and_plot(trainloader, split_name="Train")
evaluate_and_plot(testloader,  split_name="Test")
