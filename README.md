# Thai_Transformer
Thai Transformer (ThaiT) is a family of ViT models designed to automatically classify plutonic rock types in Thailand with high accuracy and generalization. The proposed models incorporate a new Error Linear Unit (new GELU) activation in the position-wise feed-forward network (FFN) and an optimized multi-head attention mechanism with merged Query (Q), Key (K), and Value (V) projections. 

<img width="697" height="762" alt="overall architectures" src="https://github.com/user-attachments/assets/df158847-d911-40ae-93a2-3833557eb26e" />

The code enables training, evaluation, and attention visualization for plutonic rock classification using plane-polarised light (PPL) and cross-polarized light (XPL) thin-section images.

# Usage
Installation

##1. Requirements
   
### Software
- Python 3.7+
- PyTorch 1.7.0 or higher
- CUDA 10.2 or higher (for GPU training)

### Tested Environment
The code has been tested with the following configuration:
- OS: Ubuntu 18.04.6 LTS
- CUDA: 10.2
- PyThorch 1.8.2
- Python 3.8.11
- GPU: NVIDIA RTX series

##2. Install all dependencies. Install pythorch, cuda and cudnn, then install other dependencies via:

```bash
pip install -r requirements.txt
```

Note: PyTorch and CUDA should be installed following the official PyTorch instructions: 
https://pytorch.org/get-started/locally/

# Dataset structure
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
