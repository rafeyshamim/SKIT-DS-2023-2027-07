# Sprint 2 — 3D CNN Architecture Design & Implementation

**Project Title:** AI-Powered 3D Medical Image Reconstruction and Disease Analysis  
**Project ID:** SKIT/DS/2023-2027/CSE-F-07  
**Team Lead:** Mohammad Rafey  
**Sprint Window:** 01-09-2026 to 30-09-2026  

---

## 1. Architectural Overview

The 3D Convolutional Neural Network (3D CNN) is designed specifically for volumetric spatial feature representation in medical imaging. The architecture leverages volumetric convolutions ($3 \times 3 \times 3$ receptive fields), batch normalization, non-linear activation, and global pooling to capture complex spatial patterns across axial, coronal, and sagittal planes simultaneously.

```
Input: (batch_size, D, H, W, 1)
   │
   ▼
[Block 1] Conv3D(f_1, (3,3,3)) + BatchNorm + ReLU + MaxPool3D(2,2,2)
   │
   ▼
[Block 2] Conv3D(f_2, (3,3,3)) + BatchNorm + ReLU + MaxPool3D(2,2,2)
   │
   ▼
[Block 3] Conv3D(f_3, (3,3,3)) + BatchNorm + ReLU + MaxPool3D(2,2,2)
   │
   ▼
[Block 4] Conv3D(f_4, (3,3,3)) + BatchNorm + ReLU (no downsampling)
   │
   ▼
[Global Pooling] GlobalAveragePooling3D -> 1D vector (f_4)
   │
   ▼
[Dense Block 1] Dense(512) + ReLU + Dropout(0.5)
   │
   ▼
[Dense Block 2] Dense(256) + ReLU + Dropout(0.5)
   │
   ▼
[Output Head] Dense(num_classes, activation="softmax")
```

---

## 2. Layer Specifications

1. **Volumetric Convolutions (`Conv3D`)**:
   - Kernel size: `(3, 3, 3)` across all blocks.
   - Padding: `'same'` to maintain boundary integrity during convolution.
   - Regularization: $L_2$ kernel regularization ($\lambda = 0.001$) to suppress overfitting on small medical cohorts.
   - Bias: Set to `use_bias=False` because `BatchNormalization` includes an affine bias parameter.

2. **Batch Normalization (`BatchNormalization`)**:
   - Stabilizes internal covariate shift across training batches.
   - Placed immediately after `Conv3D` and prior to non-linear activation.

3. **Spatial Downsampling (`MaxPooling3D`)**:
   - Strided pooling with factor `(2, 2, 2)`.
   - Selectively omitted on the terminal convolutional block to preserve feature map resolution before global pooling.

4. **Global Aggregation (`GlobalAveragePooling3D`)**:
   - Collapses spatial dimensions $(D_k, H_k, W_k)$ into a single channel vector, significantly reducing parameter count compared to flattening and providing spatial translation invariance.

5. **Regularized Classification Head**:
   - Multi-layer perceptron with dropout rates of $0.5$ and $L_2$ weight decay.
   - Final layer uses `softmax` activation yielding valid probability distributions $\sum p_i = 1.0$.

---

## 3. Configuration Decoupling

All hyper-parameters (input shape, filter depths, kernel dimensions, dropout rates, and class numbers) are dynamically injected from `config/config.yaml`.
Factory function: `src.model.architecture.get_model_from_config(config)`.
