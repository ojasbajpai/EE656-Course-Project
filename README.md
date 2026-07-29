# ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection

A PyTorch implementation of **ISTVT (Interpretable Spatial-Temporal Video Transformer)** for deepfake detection, based on the paper:

> **ISTVT: Interpretable Spatio-Temporal Video Transformer for Deepfake Detection**  
> Zhao et al., IEEE Transactions on Information Forensics and Security (2023)

This project was developed as part of the **EE656: Deep Learning** course at IIT Kanpur. The implementation reproduces the key architectural contributions of the original paper while adapting the pipeline for efficient training on a single NVIDIA T4 GPU.

---

## Overview

Most deepfake detectors rely either on:

- **Spatial artifacts**, such as blending boundaries and texture inconsistencies within individual frames, or
- **Temporal inconsistencies**, such as flickering and frame-to-frame appearance changes.

ISTVT models both simultaneously using a decomposed transformer architecture that separates spatial and temporal attention. Unlike conventional video transformers, it also introduces a **self-subtract mechanism**, allowing temporal attention to focus only on changes between consecutive frames instead of redundant static information.

The primary objective of this implementation was to faithfully reproduce these core ideas while remaining computationally feasible on limited hardware.

---

## Architecture

The overall pipeline consists of three stages:

```
Input Video
      │
      ▼
Face Detection & Cropping
      │
      ▼
EfficientNet-B0 Feature Extractor
      │
      ▼
Patch Tokenization
      │
      ▼
Spatial + Temporal CLS Tokens
      │
      ▼
Decomposed Spatial-Temporal Transformer
      │
      ▼
MLP Classification Head
      │
      ▼
Real / Fake
```

### Main Components

- Pretrained EfficientNet-B0 feature extractor
- Patch embedding and positional encoding
- Dual classification token design
- Decomposed spatial and temporal self-attention
- Self-subtract residual mechanism
- Layer-wise relevance propagation (LRP) for interpretability

---

## Implementation Details

The original paper was designed for a substantially larger training setup using multiple GPUs and the FaceForensics++ dataset. Several engineering modifications were necessary to reproduce the architecture within the constraints of Google Colab.

### Backbone Replacement

The paper trains an Xception network from scratch.

This implementation instead uses a pretrained **EfficientNet-B0** backbone from `timm`.

Reasons:

- faster convergence
- significantly lower computational cost
- better feature extraction under limited training data
- reduced GPU memory requirements

Only the feature extractor was modified; the transformer architecture remains faithful to the original design.

---

### Face Preprocessing

Each video undergoes the following preprocessing pipeline:

1. Uniform frame sampling
2. Face detection using OpenCV DNN (SSD + ResNet-10)
3. Face-centered square crop
4. Resize to 128×128
5. Cached preprocessing for faster subsequent training

Face cropping proved essential for learning meaningful forgery artifacts instead of background information.

---

### Consecutive Clip Sampling

Instead of treating an entire video as one sample, each video is converted into multiple overlapping clips.

Configuration:

- 32 cached frames per video
- clip length = 6 frames
- stride = 4

This increases the number of effective training samples while preserving temporal consistency required by the self-subtract mechanism.

---

### Decomposed Attention

Unlike standard video transformers, ISTVT performs attention in two independent stages.

**Temporal Attention**

- operates across frames
- captures temporal inconsistencies
- uses residual (self-subtracted) features

**Spatial Attention**

- operates within individual frames
- captures blending artifacts
- models spatial forgery cues

This decomposition substantially reduces computational complexity compared to full spatio-temporal attention while preserving performance.

---

### Self-Subtract Mechanism

One of the key contributions of ISTVT.

Instead of computing temporal attention directly on raw frame features,

```
Frame_t
```

the model computes

```
Frame_t - Frame_(t-1)
```

for temporal queries and keys.

This suppresses static information and forces the transformer to attend only to meaningful inter-frame changes such as flickering or inconsistent facial motion.

The original features are still used for the value projection, preserving spatial information.

---

## Training Configuration

| Parameter | Value |
|-----------|------:|
| Dataset | Celeb-DF-v2 |
| Training Videos | 944 |
| Validation Videos | 236 |
| Sequence Length | 6 Frames |
| Input Resolution | 128×128 |
| Batch Size | 8 |
| Epochs | 20 |
| Optimizer | AdamW |
| Backbone LR | 1e-5 |
| Transformer LR | 3e-4 |
| Scheduler | Linear Warmup + Cosine Decay |
| Mixed Precision | FP16 AMP |

---

## Results

### Clip-Level Performance

| Metric | Score |
|--------|------:|
| Accuracy | **93.46%** |
| AUROC | **0.9897** |

### Video-Level Performance

| Metric | Score |
|--------|------:|
| Accuracy | **95.76%** |
| AUROC | **0.9959** |

Confusion Matrix

| | Predicted Real | Predicted Fake |
|---|---:|---:|
| Actual Real | 110 | 8 |
| Actual Fake | 2 | 116 |

---

## Interpretability

Unlike many deepfake detectors, ISTVT provides visual explanations for its predictions.

Using Layer-wise Relevance Propagation (LRP), the model generates:

- **Spatial Heatmaps**
  - highlight frame-level blending artifacts
  - texture inconsistencies
  - manipulated facial regions

- **Temporal Heatmaps**
  - highlight inter-frame flickering
  - inconsistent facial motion
  - temporal forgery patterns

<Figure size 1800x900 with 18 Axes><img width="1768" height="892" alt="image" src="https://github.com/user-attachments/assets/cedca5b2-ccc7-40c2-8722-4970ef6db838" />


These visualizations make the model's predictions significantly easier to interpret.

---

## Future Improvements

Potential directions for extending this implementation include:

- Training on the full FaceForensics++ dataset
- Evaluating cross-dataset generalization
- Incorporating larger transformer backbones
- Replacing EfficientNet-B0 with pretrained Xception or ConvNeXt
- Exporting the model for real-time inference

---

## References

1. Zhao et al. **ISTVT: Interpretable Spatio-Temporal Video Transformer for Deepfake Detection**, IEEE Transactions on Information Forensics and Security, 2023.

2. Celeb-DF-v2 Dataset

3. PyTorch

4. timm (PyTorch Image Models)

5. OpenCV
