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
# Utility functions
def load_experiment(experiment_name, checkpoint_name="model_best.pt", base_dir="experiments"):
    """Load experiment details including model, config, and metrics."""
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    # Load the metrics
    metricsfile = os.path.join(outdir, 'metrics.json')
    with open(metricsfile, 'r') as f:
        metrics = json.load(f)
    # Load the model
    model = ThaiTForClassfication(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))
    return config, model, metrics

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

def visualize_attention_custom(model, test_dir, output=None, device="cuda"):
    """Visualize attention maps using a single image from test dataset."""
    model.eval()
    dataset = torchvision.datasets.ImageFolder(root=test_dir)
    classes = dataset.classes
    
    # Select 1 random image
    idx = torch.randint(len(dataset), (1,)).item()
    raw_image, label = dataset[idx]
    raw_image_np = np.asarray(raw_image)
    
    test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.44591194, 0.4460215, 0.44402866), 
                             (0.27868111, 0.28104728, 0.2775477))
    ])
    image = test_transform(raw_image).unsqueeze(0).to(device)
    model = model.to(device)
    # Ensure output attentions
    logits, attention_maps = model(image, output_attentions=True)
    prediction = torch.argmax(logits, dim=1).item()
    # Process attention maps (assumes that attention maps are available from each block)
    attention_maps = torch.cat(attention_maps, dim=1)  # shape: [B, total_heads, seq_len, seq_len]
    # We take the attention maps for the class token (position 0) and ignore the class token from key side
    attention_maps = attention_maps[:, :, 0, 1:]
    attention_maps = attention_maps.mean(dim=1)
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)
    attention_maps = F.interpolate(attention_maps.unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(raw_image_np)
    ax.imshow(attention_maps[0].detach().cpu().numpy(), alpha=0.5, cmap='jet')
    gt, pred = classes[label], classes[prediction]
    ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt == pred else "red"))
    ax.set_xticks([])
    ax.set_yticks([])
    if output is not None:
        plt.savefig(output)
        print(f"Attention map saved to {output}")
    plt.show()

# # Experiment and training hyperparameters
exp_name = 'ThaiT_B/16'
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Plot Training Results
# After training, load the experiment metrics and plot the losses and accuracies.
import matplotlib.pyplot as plt
config_loaded, model_loaded, metrics = load_experiment(exp_name)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Visualize Attention Maps
visualize_attention_custom(model_loaded, "plutonic_rocks/test", output="heat_map_01.png")
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------#
