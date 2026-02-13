# Attention visualization.py
import torchvision
import torchvision.transforms as transforms
import json
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from thait import ThaiTForClassfication

# ----------------------------------------------------------------------------------------------------------------- #
def visualize_custom_images(train_dir):
    """Visualize some training images from your custom dataset."""
    dataset = torchvision.datasets.ImageFolder(root=train_dir)
    classes = dataset.classes
    # Pick 1 random sample
    idx = torch.randint(len(dataset), (1,)).item()
    image, label = dataset[idx]

    plt.figure(figsize=(5, 5))
    plt.imshow(np.asarray(image))
    plt.title(classes[label])
    plt.xticks([])
    plt.yticks([])
    plt.show()

def visualize_attention_all(model, test_dir, output_dir, device="cuda"):
    model.eval().to(device)
    dataset = torchvision.datasets.ImageFolder(root=test_dir)
    classes = dataset.classes

    test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.44591194, 0.4460215, 0.44402866), (0.27868111, 0.28104728, 0.2775477))
    ])
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(len(dataset)):
        raw_image, label = dataset[idx]
        raw_image_np = np.asarray(raw_image)
        image = test_transform(raw_image).unsqueeze(0).to(device)

        logits, attention_maps = model(image, output_attentions=True)
        prediction = torch.argmax(logits, dim=1).item()
        
        attention_maps = torch.cat(attention_maps, dim=1)  # [B, total_heads, seq_len, seq_len]
        attention_maps = attention_maps[:, :, 0, 1:]       # cls token -> patch
        attention_maps = attention_maps.mean(dim=1)
        size = int(math.sqrt(attention_maps.size(-1)))
        attention_maps = attention_maps.view(-1, size, size)
        H, W = raw_image_np.shape[:2]
        attention_maps = F.interpolate(attention_maps.unsqueeze(1),
                                       size=(H, W),
                                       mode='bilinear',
                                       align_corners=False).squeeze(1)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(raw_image_np)
        ax.imshow(attention_maps[0].detach().cpu().numpy(), alpha=0.5, cmap="jet")
        gt, pred = classes[label], classes[prediction]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt == pred else "red"))
        ax.set_xticks([]); ax.set_yticks([])

        save_path = os.path.join(output_dir, f"att_{idx:04d}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)  # ปิด figure กัน memory leak
        print(f"Saved: {save_path}")

# # Experiment and training hyperparameters
exp_name = '/content/drive/MyDrive/experiments/ThaiT/B/16'
#----------------------------------------------------------------------------------------------------------------#
# [6]: Plot Training Results
# After training, load the experiment metrics and plot the losses and accuracies.
import matplotlib.pyplot as plt
config_loaded, model_loaded, metrics = load_experiment(exp_name)
#----------------------------------------------------------------------------------------------------------------#
# [7]: Visualize Attention Maps
visualize_attention_all(model_loaded, "/dataset/plutonic_rock_test", output_dir="/heat_map_01")
#----------------------------------------------------------------------------------------------------------------#
