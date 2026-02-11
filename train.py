import torch
import time
from torch import nn, optim
from utils import get_max_gpu_memory_gb, reset_gpu_peak_mem, save_checkpoint, save_experiment
from data import prepare_data
from model import ThaiTForClassification

config = {
    "patch_size": 16,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "image_size": 512,
    "num_classes": 15,
    "num_channels": 3,
    "qkv_bias": True,
    "use_optimizing_attention": True,
}

assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0

class Trainer:
    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device

        # histories
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.train_accuracies, self.val_accuracies, self.test_accuracies = [], [], []

        # added: throughput & memory histories
        self.throughput_hist = []
        self.mem_hist = []

    def train(self, trainloader, valloader, testloader, epochs, save_model_every_n_epochs=0):
        best_val_acc = 0.0

        for i in range(epochs):
            if torch.cuda.is_available():
                reset_gpu_peak_mem()
                torch.cuda.synchronize()
            epoch_start = time.perf_counter()
            images_this_epoch = 0
            train_loss, train_acc, images_this_epoch = self.train_epoch(trainloader)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_time = time.perf_counter() - epoch_start
            throughput = (images_this_epoch / epoch_time) if epoch_time > 0 else 0.0
            max_mem_gb = get_max_gpu_memory_gb()
            val_acc, val_loss = self.evaluate(valloader)
            test_acc, test_loss = self.evaluate(testloader)

            # ---------------- log ---------------- #
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.test_accuracies.append(test_acc)
            self.throughput_hist.append(throughput)
            self.mem_hist.append(max_mem_gb)

            print(
                f"Epoch {i+1}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | "
                f"Throughput: {throughput:.2f} img/s, Max GPU Mem: {max_mem_gb:.3f} GB"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(self.exp_name, self.model, "best")
            if save_model_every_n_epochs > 0 and (i + 1) % save_model_every_n_epochs == 0 and i + 1 != epochs:
                save_checkpoint(self.exp_name, self.model, i + 1)
                
        save_experiment(
            self.exp_name,
            config,
            self.model,
            self.train_losses,
            self.test_losses,
            self.test_accuracies,
            train_accuracies=self.train_accuracies,
            val_losses=self.val_losses,
            val_accuracies=self.val_accuracies,
            throughput_img_per_sec=self.throughput_hist,
            max_gpu_mem_GB=self.mem_hist,
        )
        
    def train_epoch(self, trainloader):
        self.model.train()
        total_loss, correct = 0.0, 0
        images_seen = 0

        for images, labels in trainloader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits, _ = self.model(images)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()
            bsz = images.size(0)
            images_seen += bsz
            total_loss += loss.item() * bsz
            correct += (logits.argmax(dim=1) == labels).sum().item()
        avg_loss = total_loss / images_seen
        acc = correct / images_seen
        
        return avg_loss, acc, images_seen

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss, correct = 0.0, 0
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits, _ = self.model(images)
            loss = self.loss_fn(logits, labels)
            total_loss += loss.item() * len(images)
            correct += (logits.argmax(dim=1) == labels).sum().item()
        return correct / len(dataloader.dataset), total_loss / len(dataloader.dataset)

def main(exp_name, batch_size, epochs, lr, save_model_every, device):
    trainloader, valloader, testloader, _ = prepare_data(batch_size=batch_size)
    model = ThaiTForClassification(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, exp_name, device=device)
    trainer.train(trainloader, valloader, testloader, epochs, save_model_every_n_epochs=save_model_every)

device = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------------------------------------------------------------------- #
