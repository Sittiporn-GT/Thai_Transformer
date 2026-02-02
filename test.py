import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, precision_score

from model import ThaiForClassification

# Load Experiment Metrics and Plot Curves
exp_name = 'experiments_ViT_test_LT'
exp_dir = os.path.join("experiments", exp_name)

# Load saved metrics
metrics_path = os.path.join(exp_dir, 'metrics.json')
with open(metrics_path, 'r') as f:
    metrics = json.load(f)
train_losses = metrics.get("train_losses", [])
val_losses = metrics.get("val_losses", [])
test_losses = metrics.get("test_losses", [])
val_accuracies = metrics.get("val_accuracies", [])
train_accuracies = metrics.get("train_accuracies", [])

# Load the Custom ViT Model

def load_model(exp_name, checkpoint_name="model_best.pt", base_dir="experiments"):
    exp_dir = os.path.join(base_dir, exp_name)
    configfile = os.path.join(exp_dir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    
    model = ThaiForClassification(config)
    cpfile = os.path.join(exp_dir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile, map_location='cpu'))
    return model, config

model, config = load_model(exp_name, checkpoint_name="model_best.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Define Data Transformers and Load Datasets
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.50222724, 0.50163647, 0.49506611), 
                         (0.29474634, 0.29574405, 0.29541864))
])

# Load datasets
train_dir = "plutonic_rocks_split_05/train"
train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)

test_dir = "plutonic_rocks_split_05/test"
test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Compute and Plot Confusion Matrices
def compute_metrics(data_loader, dataset_name):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            logits, _ = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data_loader.dataset.classes, yticklabels=data_loader.dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({dataset_name})')
    plt.show()
    
    mean_acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    print(f"Classification Report ({dataset_name}):")
    print(classification_report(all_labels, all_preds, target_names=data_loader.dataset.classes))
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}\n")

# Generate Metrics for Train & Test Sets
compute_metrics(train_loader, "Train Set")
compute_metrics(test_loader, "Test Set")

# Plot Loss Curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch') 
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.show()

# Plot Accuracy Curves
plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.legend()
plt.show()