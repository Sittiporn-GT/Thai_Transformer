import os
import json
import torch
from data import prepare_data
from model import ThaiTForClassification
from utils import evaluate_and_plot

exp_dir = "/content/drive/MyDrive/experiments/ThaiT/B/16"
checkpoint_path = os.path.join(exp_dir, "model_best.pt")
config_path = os.path.join(exp_dir, "config.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
with open(config_path, "r") as f:
    config = json.load(f)
model = ThaiTForClassification(config).to(device)

ckpt = torch.load(checkpoint_path, map_location=device)
state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
model.load_state_dict(state, strict=True)
model.eval()

trainloader, _, testloader, classes = prepare_data(batch_size=64)
evaluate_and_plot(trainloader, split_name="Train")
evaluate_and_plot(testloader,  split_name="Test")
