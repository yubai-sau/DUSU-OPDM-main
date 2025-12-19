# UnmixNet + CDnet: Hyperspectral Change Detection Framework

A PyTorch implementation of a unified framework combining UnmixNet (spectral unmixing) and CDnet (change detection) for hyperspectral image analysis and change detection tasks.

## Overview

This repository implements a two-stage approach for hyperspectral change detection:
1. **UnmixNet**: Performs spectral unmixing using gradient descent unfolding with proximal refinement
2. **CDnet**: Detects changes using a siamese encoder with  classifier

The framework is particularly designed for agricultural monitoring applications, such as farmland change detection using multi-temporal hyperspectral data.

## Key Features

- **Spectral Unmixing**: UnmixNet decomposes hyperspectral images into endmembers and abundance maps
- **Change Detection**: CDnet identifies changes between temporal image pairs
- **Multi-temporal Analysis**: Supports analysis of bi-temporal hyperspectral datasets
- **Flexible Architecture**: Configurable through YAML configuration files
- **GPU Acceleration**: Full PyTorch implementation with CUDA support

## Architecture

### UnmixNet
- **Endmember Extraction**: Learns global endmembers `E ∈ R^{C×P}`
- **Abundance Estimation**: Computes abundance maps `A ∈ R^{P×H×W}`
- **Gradient Descent Unfolding**: K-stage iterative refinement with proximal operators
- **Simplex Projection**: Ensures non-negative and sum-to-one constraints

### CDnet
- **Siamese Encoder**: Shared feature extraction for temporal images
- **Feature Fusion**: Combines temporal features for change analysis

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.1+
- CUDA (recommended for GPU acceleration)


# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
torch>=2.1
torchvision>=0.16
numpy>=1.23
scikit-learn>=1.3
PyYAML>=6.0
tqdm>=4.65
matplotlib>=3.7
```

## Data Preparation

### Input Data Format
The framework expects NumPy arrays (`.npy` files) with the following structure:

- **T1.npy**: First temporal image `(H, W, C)`
- **T2.npy**: Second temporal image `(H, W, C)`
- **label.npy**: Ground truth change map `(H, W)`
- **mask.npy** (optional): Valid pixel mask `(H, W)`

### Configuration
Update the data path in the configuration file:

```yaml
# configs/farmland.yaml
data:
  root: "your/data/path"  # Update this path
  t1: T1.npy
  t2: T2.npy
  label: label.npy
  mask: null  # or "mask.npy" if available
```

## Usage

### Training
```bash
# Pre-training stage (spectral unmixing)
python train.py --config configs/farmland.yaml --stage pretrain

# Joint training stage (unmixing + change detection)
python train.py --config configs/farmland.yaml --stage joint
```



## Contributing

We welcome contributions to improve the framework! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation




