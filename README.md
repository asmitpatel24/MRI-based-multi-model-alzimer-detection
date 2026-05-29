Repo. -> Master Branch
# Alzheimer's Disease MRI Classifier (SOTA Hybrid Architecture)

Detect Alzheimer's disease stages from brain MRI scans using a state-of-the-art **Hybrid ConvNeXt + Swin Transformer** deep learning model. The project leverages a robust consortium of datasets including **ADNI, OASIS-3, and AIBL** for extensive training and robust generalization.

## Key Features

- **Hybrid Architecture:** Fuses the spatial detail of **ConvNeXt (CNN)** with the global contextual awareness of **Swin Transformer**.
- **Modern Tech Stack:** Built with **PyTorch**, **FastAPI**, and **Next.js**.
- **Interactive Dashboard:** Modern web interface for image uploads and real-time inference.
- **GPU Accelerated:** Optimized for NVIDIA GPUs via CUDA.

---

## Architecture & Models

The model implements a dual-branch feature extraction strategy:

1.  **CNN Branch (ConvNeXt-Tiny):** A modern convolutional neural network that adopts Transformer-inspired design improvements. It is exceptionally good at capturing local textures and edge details in medical imaging.
2.  **Transformer Branch (Swin Transformer Tiny):** A hierarchical Vision Transformer that uses shifted windows to model multi-scale global context.
3.  **Fusion Head:** Combines features from both branches (1536-dimensional vector) through a dense MLP (Multi-Layer Perceptron) for final classification.

**Target Classes:**
- `NonDemented`
- `VeryMildDemented`
- `MildDemented`
- `ModerateDemented`

---

## Technology Stack

### Artificial Intelligence & Machine Learning
- **PyTorch:** Core deep learning framework.
- **torchvision:** Image processing and augmentation.
- **timm (Torch Image Models):** Library for state-of-the-art vision backbones.
- **scikit-learn:** Used for classification metrics and evaluation.
- **Matplotlib / Seaborn:** Used for generating training curves and confusion matrices.
- **NumPy / Pillow:** Data handling and image manipulation.

### Backend (API)
- **FastAPI:** High-performance asynchronous REST API.
- **Uvicorn:** ASGI server for production-grade serving.
- **Python-Multipart:** For handling file uploads.

### Frontend (Dashboard)
- **Next.js 15+:** Modern React framework (App Router).
- **TypeScript:** Type-safe frontend development.
- **Tailwind CSS v4:** Advanced styling with the latest utility-first features.
- **Lucide React:** Premium icon set.
- **Framer Motion:** Smooth UI animations.

---

## Project Structure

```
alzimer model/
├── app.py              # FastAPI Backend
├── hybrid_model.py     # PyTorch Model Architecture (ConvNeXt + Swin)
├── train.py            # Advanced 2-Phase Training Pipeline
├── dataset.py          # PyTorch Dataset & Data Augmentation
├── predict.py          # CLI Prediction Tool
├── alzheimer_hybrid_model.pth # Trained Model Weights (PyTorch)
├── start_dashboard.bat # One-click script to start Frontend + Backend
└── frontend/           # Next.js Dashboard Source Code
```

---

## Getting Started

### 1. Requirements
Ensure you have Python 3.13 and Node.js installed.
```bash
# Install Python dependencies
pip install torch torchvision timm fastapi uvicorn pillow scikit-learn matplotlib seaborn tqdm
```

### 2. Run the Dashboard
Simply double-click `start_dashboard.bat` (Windows) or:
```bash
# Terminal 1: Backend
python app.py

# Terminal 2: Frontend
cd frontend
npm run dev
```
Visit **http://localhost:3000** to use the dashboard.

### 3. Training
To retrain the model on your own data:
```bash
python train.py --epochs 10 --ft-epochs 20 --batch-size 8
```

---

## Results & Performance
The training pipeline automatically generates:
- `training_history.png`: Accuracy/Loss curves.
- `confusion_matrix.png`: Per-class model performance.

---

> ⚠️ **Disclaimer**: This tool is for research and educational purposes only. It should not be used for clinical diagnosis. Always consult a medical professional.
