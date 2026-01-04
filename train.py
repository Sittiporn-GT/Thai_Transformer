# train.py
import torch
from torch import nn, optim
from utilities import save_experiment, save_checkpoint
from data import prepare_data
from thait import ThaiTForClassfication

# ----------------------------------------------------------------------------------------------------------------- #
# Model configuration (hyperparameters)
config = {
    "patch_size": 16,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 4 * 768,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "image_size": 512,
    "num_classes": 15,
    "num_channels": 3,
    "qkv_bias": True,
    "use_optimizing_attention": True,
}

# ----------------------------------------------------------------------------------------------------------------- #
# Checks for configuration errors
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0

# ----------------------------------------------------------------------------------------------------------------- #
# Trainer class: handles training loop, validation/test evaluation, checkpointing
class Trainer:
    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device

    def train(self, trainloader, valloader, epochs, save_model_every_n_epochs=0):
        best_val_acc = 0.0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for i in range(epochs):
            train_acc, train_loss = self.train_epoch(trainloader)
            val_acc, val_loss = self.evaluate(valloader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Print training and validation metrics
            print(f"Epoch {i+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Train Acc: {train_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(self.exp_name, self.model, "best")
            
            if save_model_every_n_epochs > 0 and (i + 1) % save_model_every_n_epochs == 0 and i + 1 != epochs:
                save_checkpoint(self.exp_name, self.model, i + 1)
        
        save_experiment(self.exp_name, config, self.model, train_losses, val_losses, train_accs, val_accs)

    def train_epoch(self, trainloader):
        self.model.train()
        total_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(images)[0], labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss, correct = 0, 0
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits, _ = self.model(images)
            loss = self.loss_fn(logits, labels)
            total_loss += loss.item() * len(images)
            correct += (logits.argmax(dim=1) == labels).sum().item()
        return correct / len(dataloader.dataset), total_loss / len(dataloader.dataset)
    
# ----------------------------------------------------------------------------------------------------------------- #
def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Training hyperparameters
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--save-model-every", type=int, default=0)
    args = parser.parse_args()
    
    trainloader, valloader, _ = prepare_data(batch_size=args.batch_size)
    model = ThaiTForClassfication(config)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device=args.device)
    trainer.train(trainloader, valloader, args.epochs, save_model_every_n_epochs=args.save_model_every)

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------------------------------------------- #
