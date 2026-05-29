# 📑 Technical Documentation: Alzheimer Hybrid Model

## 1. Executive Summary
This project implements a state-of-the-art (SOTA) hybrid deep learning system for the classification of Alzheimer’s Disease from MRI scans. By combining **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)**, the system achieves a robust balance between local feature detection (atrophy, texture) and global structural awareness.

### 1.1 Data Source: ADNI + OASIS-3 + AIBL
The primary dataset used is a comprehensive combination of three major neuroimaging initiatives:
1. **ADNI (Alzheimer's Disease Neuroimaging Initiative):** A massive longitudinal study providing high-quality multicenter MRI scans.
2. **OASIS-3 (Open Access Series of Imaging Studies):** A retrospective compilation providing a robust mix of normal aging and cognitive decline cases.
3. **AIBL (Australian Imaging, Biomarker & Lifestyle Flagship Study of Ageing):** Extends geographical diversity and provides complementary high-resolution imaging data.

These datasets are merged and categorized into four clinical stages: Non-Demented, Very Mild, Mild, and Moderate Alzheimer’s.

## 2. Model Specifications

### 2.1 Branch A: ConvNeXt-Tiny
- **Type:** Convolutional Neural Network (CNN)
- **Rationale:** ConvNeXt is designed to bridge the gap between ResNets and Transformers. It uses large kernels and depth-wise convolutions to provide superior spatial feature extraction relevant to medical imaging.
- **Output:** 768-dimensional feature vector.

### 2.2 Branch B: Swin Transformer Tiny
- **Type:** Hierarchical Vision Transformer
- **Rationale:** Unlike vanilla ViT, Swin uses shifted windows to capture hierarchical representations. This is critical for brain MRI, where structural anomalies can appear at multiple scales.
- **Output:** 768-dimensional feature vector.

### 2.3 Fusion Strategy
- **Mechanism:** Late Fusion.
- **Process:** The 768d vectors from both branches are concatenated into a **1536d** global feature vector.
- **MLP Head:** A dense network consisting of Linear layers, Batch Normalization, and Dropout (0.4) reduces the features and outputs logits for 4 classes.

## 3. Computational Tools & Libraries

### Core Intelligence
| Tool | Version | Purpose |
|------|---------|---------|
| **PyTorch** | 2.6.0+ | Tensor computations and backpropagation. |
| **torchvision** | Latest | Image transformations and preprocessing. |
| **timm** | Latest | Access to pre-trained ConvNeXt and Swin weights. |
| **CUDA** | 12.4 | GPU acceleration for training and inference. |

### Data Science & Analysis
- **NumPy:** Numerical array manipulation.
- **PIL (Pillow):** Image I/O and resizing.
- **Scikit-learn:** Used for `classification_report` and `balanced_accuracy_score`.
- **Matplotlib/Seaborn:** Data visualization.

### Web Engine (Dashboard)
- **FastAPI:** Selected for its asynchronous capabilities and speed in handling image uploads.
- **Next.js 15:** Selected for its server-side rendering and modern component-level architecture.
- **Tailwind CSS v4:** Modern styling with full HSL color support and JIT compilation.

## 4. Training Methodology

### 4.1 Two-Phase Strategy
1.  **Phase 1 (Frozen):** Encoders are frozen; training focuses exclusively on the fusion head to align feature distributions.
2.  **Phase 2 (Fine-tuning):** The deepest layers of both ConvNeXt and Swin are unfrozen, allowing the model to adapt specifically to the MRI domain.

### 4.2 Optimization
- **Optimizer:** AdamW (Weight Decay: 1e-4).
- **Loss:** Weighted Cross-Entropy (compensates for class imbalance).
- **Scheduler:** Cosine Warmup Scheduler for stable convergence.

## 5. Deployment & Integration
The system is divided into a microservices-style architecture:
- **`app.py`**: A stateless API that receives images and returns JSON results.
- **Dashboard**: A React-based interface that communicates with the API via Fetch.

---
*Drafted on: February 19, 2026*
