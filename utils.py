import json, os, torch
import datetime

def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def get_max_gpu_memory_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return None

def reset_gpu_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f"model_{epoch}.pt")
    torch.save(model.state_dict(), cpfile)
    return cpfile

def save_experiment(
    experiment_name, config, model, train_losses, test_losses, test_accuracies, train_accuracies=None,
    val_losses=None, val_accuracies=None, throughput_img_per_sec=None, max_gpu_mem_GB=None,
    base_dir="experiments"
):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, sort_keys=True, indent=4)

    data = {
        "timestamp": now_iso(),
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "throughput_img_per_sec": throughput_img_per_sec,
        "max_gpu_mem_GB": max_gpu_mem_GB,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(data, f, sort_keys=True, indent=4)
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)

def load_experiment(
    experiment_name,
    checkpoint_name="model_final.pt",
    base_dir="experiments",
    model_class=None
):
    outdir = os.path.join(base_dir, experiment_name)

    with open(os.path.join(outdir, "config.json"), "r") as f:
        config = json.load(f)

    with open(os.path.join(outdir, "metrics.json"), "r") as f:
        data = json.load(f)

    train_losses = data.get("train_losses")
    test_losses = data.get("test_losses")
    test_accuracies = data.get("test_accuracies")
    train_accuracies = data.get("train_accuracies")
    val_losses = data.get("val_losses")
    val_accuracies = data.get("val_accuracies")
    throughput_img_per_sec = data.get("throughput_img_per_sec")
    max_gpu_mem_GB = data.get("max_gpu_mem_GB")

    model = None
    if model_class is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_class(config).to(device)

        cpfile = os.path.join(outdir, checkpoint_name)
        try:
            state = torch.load(cpfile, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(cpfile, map_location=device)

        model.load_state_dict(state)
        model.eval()

    return (config, model, train_losses, test_losses, test_accuracies, train_accuracies, val_losses, val_accuracies, throughput_img_per_sec, max_gpu_mem_GB)
