# Modular PyTorch Vision Pipeline & Benchmark: EuroSAT

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-blueviolet.svg)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, modular computer vision pipeline implemented in pure PyTorch evaluating land-use classification on Sentinel-2 satellite imagery from the **EuroSAT** dataset (27,000 images, 10 spectral classes, 64x64 resolution).

Rather than relying on monolithic tutorial notebooks, this repository provides a decoupled, production-style architecture featuring automated Bayesian Hyperparameter Optimization (Optuna) and a three-stage controlled ablation benchmark.

---

## Benchmark Results

All configurations were trained across 10 epochs and evaluated on an identical, strictly held-out test split (N = 4,050).

| Stage | Model Architecture | Training Strategy | Parameters | Val Acc (Best) | Test Acc |
|---|---|---|---|---|---|
| **Stage 1** | SimpleCNN (3-block) | Scratch (No Augmentation, Optuna HPO) | 1,085,066 | 86.94% | **86.99%** |
| **Stage 2** | SimpleCNN (3-block) | Scratch (Augmentation: Flips + Rotations) | 1,085,066 | 83.53% | **83.09%** |
| **Stage 3** | ResNet-18 | Linear Probe / Frozen Backbone | 5,130 (11.1M total) | 80.94% | **79.80%** |

---

## Technical Analysis & Empirical Findings

### 1. Bayesian Optimization Impact
Manual tuning often leads to suboptimal convergence or unstable training dynamics. Using **Optuna** with a **Tree-structured Parzen Estimator (TPE)** and dynamic **Median Pruning** over 5 trials, the search space converged on:
* **Optimizer**: `SGD` with momentum (0.9)
* **Learning Rate**: 7.53e-4
* **Classifier Dropout**: 0.414

This configuration allowed `SimpleCNN` to surpass **82.5% validation accuracy in just 3 epochs**, eventually reaching **86.99% test accuracy**.

### 2. Regularization vs. Convergence Speed
* **Stage 1 (Baseline)** reached 86.99% test accuracy rapidly, benefiting from a fixed spatial distribution.
* **Stage 2 (Augmentation)** introduced spatial variance via random horizontal flips and +/- 15 deg affine rotations. While this regularizes against spatial memorization, training loss decayed at a slower rate (0.4338 vs. 0.3096), indicating that heavily augmented models require a 20-30 epoch window to outperform unaugmented counterparts.

### 3. Remote Sensing Domain Shift & Linear Probing
Pretrained ResNet-18 converged rapidly (reaching 76% in epoch 1), but stalled near ~80% test accuracy. Because only the classification head was trained (`fc = nn.Linear(512, 10)`), the frozen lower blocks relied on low-level terrestrial perspective filters optimized on ImageNet (natural objects, perspective geometry). Satellite imagery is nadir (top-down) and rotational-invariant, demonstrating the domain gap in naive transfer learning and highlighting the need for full or residual block (`layer4`) fine-tuning.

---

## Pipeline Architecture

The system is decoupled into modular functional units under `src/`:

eurosat-cnn-transfer-learning/
├── src/
│   ├── dataset.py       # Deterministic 70/15/15 split, PyTorch DataLoaders & dynamic TransformSubset
│   ├── model.py         # Custom 3-block SimpleCNN & transfer-ready ResNet-18 definitions
│   ├── train.py         # Standardized 5-step PyTorch forward/backward optimization loop
│   ├── evaluate.py      # Inference routine (torch.no_grad, eval mode, metric accumulation)
│   ├── tune.py          # Optuna HPO objective, parameter samplers, and trial pruning
│   └── experiment.py    # Multi-stage driver running validation-based checkpointing
├── checkpoints/         # Serialized model state dictionaries (.pth)
└── README.md

---






