# FaceClass: Real-Time Facial Gender Classification

A practical, real-time facial gender classification system that combines deep representation learning with classical convex optimization. Built with **PyTorch (ResNet-18)**, an **RBF-Kernel Support Vector Machine (SVM)**, **OpenCV YuNet**, and a lightweight **FastAPI** web interface.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-91.98%25-brightgreen.svg)]()
[![Balanced Accuracy](https://img.shields.io/badge/Balanced_Accuracy-91.95%25-brightgreen.svg)]()

---

## What is FaceClass?

Most modern facial attribute classifiers train an end-to-end neural network with a softmax classification head. While effective, standard softmax outputs can overfit on nuanced boundary samples and often require heavy GPU compute during inference.

**FaceClass** takes a hybrid two-stage approach:
1. **Deep Feature Extraction (ResNet-18)**: Instead of treating the CNN as the final decision maker, we use a fine-tuned ResNet-18 backbone as a rich feature extractor. It maps any detected face crop into a compact 512-dimensional embedding vector.
2. **Maximum-Margin Classification (RBF-SVM)**: An RBF-kernel Support Vector Machine takes those 512-dimensional embeddings and finds the mathematically optimal maximum-margin hyperplane separating classes. This yields strong generalization and balanced recall across genders.
3. **Fast Face Detection (OpenCV YuNet)**: An ONNX-based YuNet face detector localizes single and multi-face scenes in under 20ms on a standard CPU.

The entire pipeline runs smoothly at **< 35 ms per frame on a regular CPU**—no dedicated GPU required.

---

## Key Highlights

- **Hybrid CNN + SVM Pipeline**: Leverages the representation strength of deep convolutional layers alongside the robustness of maximum-margin classification.
- **Fair & Balanced**: Class-weighted optimization keeps recall balanced between male (92.26%) and female (92.48%) faces, avoiding one-sided bias.
- **Fast CPU Inference**: Sub-35ms total latency makes it practical for live webcam streams and laptop webcams.
- **Interactive Web App**: Includes a built-in web frontend with both drag-and-drop image analysis and live webcam detection.
- **Project Presentation Included**: Full slide deck and presentation document are available directly in [`docs/`](docs/).

---

## Architecture Overview

```
 ┌─────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
 │   Input Image   │ ───>  │  OpenCV YuNet ONNX   │ ───>  │  Contextual Cropping   │
 │ (Webcam / File) │       │    Face Detection    │       │     (+15% Padding)     │
 └─────────────────┘       └──────────────────────┘       └────────────────────────┘
                                                                       │
                                                                       ▼
 ┌─────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
 │  Gender Output  │ <───  │    RBF-Kernel SVM    │ <───  │   ResNet-18 Backbone   │
 │ (Label & Conf)  │       │  Decision Boundary   │       │   (512-D GAP Vector)   │
 └─────────────────┘       └──────────────────────┘       └────────────────────────┘
```

### Flow
1. **Detection & Alignment**: YuNet detects faces with confidence thresholding and bounding boxes padded by 15% to retain facial context (hairline, jawline).
2. **Embedding**: Face crops are resized to 224x224, normalized via ImageNet statistics, and passed through ResNet-18 up to the Global Average Pooling layer ($\mathbb{R}^{512}$).
3. **Classification**: Embeddings are scaled via `StandardScaler` and classified by the RBF Support Vector Machine.

---

## Evaluation & Results

Evaluated on the held-out test split of the **UTKFace** benchmark dataset:

| Metric | Score | Note |
|---|---|---|
| **Overall Accuracy** | **91.98%** | Correct classifications across test set |
| **Balanced Accuracy** | **91.95%** | Average recall across both classes |
| **Macro F1-Score** | **0.9196** | Harmonic mean of precision and recall |
| **Male Recall** | **92.26%** | Correctly identified male faces |
| **Female Recall** | **92.48%** | Correctly identified female faces |
| **Recall Disparity ($\Delta$)** | **0.22%** | Negligible gap between genders |

### Best Hyperparameters:
- **CNN Backbone**: ResNet-18 fine-tuned on `layer4` + `fc` (AdamW, lr = $10^{-4}$, Dropout = 0.45)
- **Feature Vector**: 512 dimensions (Global Average Pooling)
- **SVM Kernel**: Radial Basis Function (RBF)
- **Regularization ($C$)**: 0.01
- **Gamma ($\gamma$)**: `'scale'` ($1 / (d \cdot \sigma^2)$)
- **Class Weights**: `'balanced'`

---

## Project Presentations & Documentation

The project slide deck and documentation are stored in the [`docs/`](docs/) folder:
- **Presentation Slides (PowerPoint)**: [`docs/AIProject_SVM.pptx`](docs/AIProject_SVM.pptx)
- **Presentation Document (PDF)**: [`docs/AIProject_SVM.pdf`](docs/AIProject_SVM.pdf)

---

## Quick Start Guide

### 1. Clone the Repository

```bash
git clone git@github.com:subeshyadav3/AI_Project_SVM.git
cd AI_Project_SVM
```

### 2. Set Up a Virtual Environment

Using standard Python `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

*Or using [uv](https://github.com/astral-sh/uv) (recommended for fast setup):*
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r backend/requirements.txt
```

> **Note**: For training and notebooks, install the full requirements using `pip install -r requirements.txt`.

### 3. Run the Web Application

Start the FastAPI server:
```bash
python backend/app.py
```

Once running, open your browser at:
```
http://localhost:8000
```

- **Image Analysis**: Upload or drag-and-drop any image to view face bounding boxes, predicted gender, and confidence ratings.
- **Live Webcam**: Stream video from your camera with real-time detection overlays (Blue = Male, Pink = Female).

---

## Training from Scratch (Optional)

If you'd like to train your own models or experiment with different SVM kernels:

1. **Dry run test** (quick sanity check on synthetic samples):
   ```bash
   python train.py --dry-run
   ```

2. **Full training pipeline** on the UTKFace dataset:
   ```bash
   python train.py --data-dir /path/to/UTKFace --epochs 10 --batch-size 32
   ```

3. **Interactive Notebook**:
   ```bash
   jupyter notebook notebooks/SVM_GENDER_TRAINING.ipynb
   ```

---

## Repository Structure

```
.
├── backend/
│   ├── app.py                     # FastAPI backend and static file server
│   ├── face_detector.py           # OpenCV YuNet face detection & coordinate logic
│   ├── gender_model.py            # ResNet-18 feature extractor & SVM predictor
│   └── requirements.txt           # Minimal backend dependencies
├── frontend/
│   ├── index.html                 # Clean, responsive web UI
│   ├── script.js                  # Webcam stream handler, canvas drawing & API client
│   └── style.css                  # UI styling and themes
├── models/
│   ├── face_detection_yunet_2023mar.onnx  # Pretrained YuNet ONNX weights
│   ├── gender_resnet18_best.pth           # Fine-tuned ResNet-18 weights
│   └── gender_complete_model.joblib       # Bundled SVM model + StandardScaler + metadata
├── notebooks/
│   └── SVM_GENDER_TRAINING.ipynb          # End-to-end training and SVM grid search notebook
├── results/
│   ├── dataset_split.csv                  # Stratified train/val/test splits
│   ├── gender_cnn_history.csv             # Loss & accuracy history
│   └── gender_svm_search.csv              # SVM hyperparameter search log
├── docs/
│   ├── AIProject_SVM.pdf                  # Presentation document (PDF)
│   └── AIProject_SVM.pptx                 # Presentation slide deck (PPTX)
├── train.py                               # Training, feature extraction & grid search pipeline
├── requirements.txt                       # Full project dependencies (training + backend)
└── README.md                              # This file
```

---

## License & Notes

Developed for academic research and evaluation in facial attribute analysis using hybrid deep learning and convex optimization techniques. Feel free to use and adapt this project for educational and experimental purposes!
