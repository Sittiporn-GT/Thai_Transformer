# Thai_Transformer
Thai Transformer (ThaiT) is a family of ViT models designed to automatically classify plutonic rock types in Thailand with high accuracy and generalization. The proposed models incorporate a new Error Linear Unit (new GELU) activation in the position-wise feed-forward network (FFN) and an optimized multi-head attention mechanism with merged Query (Q), Key (K), and Value (V) projections. 

<img width="697" height="762" alt="overall architectures" src="https://github.com/user-attachments/assets/df158847-d911-40ae-93a2-3833557eb26e" />

The code enables training, evaluation, and attention visualization for plutonic rock classification using plane-polarised light (PPL) and cross-polarized light (XPL) thin-section images.

# Usage

### Installation

1. Requirements

- Python 3.7+
- PyTorch 1.7.0 or higher
- CUDA 10.2 or higher (for GPU training)

The code has been tested with the following configuration:
- OS: Ubuntu 18.04.6 LTS
- CUDA: 10.2
- PyThorch 1.8.2
- Python 3.8.11
- GPU: NVIDIA RTX series

2. Install all dependencies. Install pythorch, cuda and cudnn, then install other dependencies via:

```bash
pip install -r requirements.txt
```

Note: PyTorch and CUDA should be installed following the official PyTorch instructions: 
https://pytorch.org/get-started/locally/


### Dataset structure
Organize the dataset folder in the following structure:

```bash
data/
├── train/
│   ├── Granite/
          ├──01_granite_XPL
          ├──01_granite_PPL
          ├──02_granite_XPL
          ...
│   ├── Gabbro/
│   └── ...
├── val/
│   ├── Granite/
│   └── ...
├── test/
│   ├── Granite/
│   └── ...
└── demo_test/
    ├── Granite/
    ├── Diorite/
    └── ...
```

Images should be organized following the standard ImageFolder format used by torchvision.

### Training 

To train the Thai Transformer model from scratch:

```bash
python train.py \
  --exp-name thait_base \
  --batch-size 256 \
  --epochs 100 \
  --lr 0.01
```

The script reports classification accuracy and loss on the test set.

### Evaluation 

To evaluate a trained model on the dataset:

```bash
python evaluate.py \
  --exp-name thait_base \
  --checkpoint model_best.pt
```

### Attention Visualization

To visualize self-attention maps and Grad-CAM-based interpretations:

```bash
python visualize_attention.py \
  --exp-name thait_base \
  --data-dir data/test \
  --output attention_results.png
```

This script overlays attention heatmaps on thin-section, highlighting petrographic features such as twinning and extinction patterns.


### Reproducibility

- All experiments reported in the manuscript were conducted using the configuration files provided in the repository.
- Random seeds can be fixed in the training script to ensure deterministic behavior.
- The demo dataset allows verification of code execution and output consistency.

# Result

We offer pre-trained weights for different parameter models on the same plutonic rock datasets.

### Available Model Weights

| Model | Patch Size | Training | Weight |
|------|-----------|----------|--------|
| ThaiT-Base | 16×16 | Scratch | [ThaiT-Base](https://github.com/USERNAME/Thai-Transformer/releases/download/v1.0.0/thait-bese.pt) |
| ThaiT-Large | 16×16 | Scratch | [ThaiT-Large](https://github.com/USERNAME/Thai-Transformer/releases/download/v1.0.0/thait-large.pt) |
