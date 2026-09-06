# FaceClass: Facial Gender Classification

> **Robust Real-Time Facial Gender Classification using Deep CNN Feature Extraction (ResNet-18) and Support Vector Machines (RBF Kernel)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-91.98%25-brightgreen.svg)]()
[![Balanced Accuracy](https://img.shields.io/badge/Balanced_Accuracy-91.95%25-brightgreen.svg)]()

---

## 1. Project Overview

**FaceClass** is an end-to-end computer vision system designed for accurate, real-time facial gender classification from in-the-wild photography and live webcam streams.

Rather than relying purely on an end-to-end deep neural network with a softmax output, FaceClass implements a **hybrid two-stage architecture**:
1. **Deep Feature Representation**: A fine-tuned **ResNet-18** convolutional network extracts a compact, highly expressive 512-dimensional semantic embedding vector from cropped face images.
2. **Maximum-Margin Classification**: A **Soft-Margin Support Vector Machine (SVM) with a Radial Basis Function (RBF) Kernel** separates the feature space with a mathematically optimal maximum margin, delivering superior generalization and preventing overfitting.
3. **Real-Time Edge Detection**: An anchor-based **OpenCV YuNet ONNX** face detector provides sub-20ms face localization, supporting multi-face scenes and live camera feeds.

---

## 2. Key Highlights & Contributions

- **Orientation & Tilt Robustness**: Trained with rotational ($\pm 15^\circ$), affine, and perspective augmentations to eliminate tilt sensitivity.
- **Symmetric Gender Precision**: Employs class-weighted margin penalties to equalize recall between male (92.26%) and female (92.48%) faces, eliminating gender bias.
- **Fast CPU Inference**: End-to-end processing latency of **< 35 ms per image** on standard consumer CPUs without requiring a dedicated GPU.
- **Presentation-Ready Web UI**: Built-in interactive web application featuring drag-and-drop image analysis and live webcam feed.
- **Comprehensive Theoretical Derivations**: Complete mathematical proofs covering Lagrangian duality, KKT conditions, and RBF Hilbert space projection in the [`docs/`](docs/) directory.

---

## 3. System Architecture

```
┌─────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│  Input Image    │ ───>  │  OpenCV YuNet ONNX   │ ───>  │  Contextual Cropping   │
│ (Webcam / File) │       │    Face Detection    │       │     (+15% Padding)     │
└─────────────────┘       └──────────────────────┘       └────────────────────────┘
                                                                     │
                                                                     ▼
┌─────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│  Gender Output  │ <───  │    RBF-Kernel SVM    │ <───  │   ResNet-18 Backbone   │
│ (Label & Conf)  │       │  Decision Boundary   │       │   (512-D GAP Vector)   │
└─────────────────┘       └──────────────────────┘       └────────────────────────┘
```

### The Two-Stage Mathematical Formulation:
$$\mathbf{x} \xrightarrow{\text{ResNet-18 } \phi(\cdot)} \mathbf{z} \in \mathbb{R}^{512}
  \xrightarrow{\text{StandardScaler}} \tilde{\mathbf{z}} \in \mathbb{R}^{512}
  \xrightarrow{\text{RBF-SVM } f(\cdot)} \hat{y} = \operatorname{sign}(f(\tilde{\mathbf{z}}))$$

---

## 4. Quantitative Evaluation Results

Evaluation performed on the held-out test split of the **UTKFace** benchmark dataset:

| Metric | Score | Description |
|---|---|---|
| **Overall Accuracy** | **91.98%** | Percentage of test samples classified correctly |
| **Balanced Accuracy** | **91.95%** | Unweighted mean of recall across both genders |
| **Macro F1-Score** | **0.9196** | Harmonic mean of precision and recall |
| **Male Recall** | **92.26%** | Accurate classification of male subjects |
| **Female Recall** | **92.48%** | Accurate classification of female subjects |
| **Recall Disparity ($\Delta$)** | **0.22%** | Symmetrical performance with negligible bias |

### Optimal Hyperparameters (Grid Search):
- **Backbone**: ResNet-18 fine-tuned on `layer4` + `fc` (Dropout = 0.45, AdamW, $\text{lr} = 10^{-4}$)
- **Feature Dimension**: 512
- **Kernel**: Radial Basis Function (RBF)
- **Regularization ($C$)**: $0.01$
- **Kernel Coefficient ($\gamma$)**: `'scale'` ($1 / (d \cdot \sigma^2)$)
- **Class Weights**: `balanced`

---

## 5. Repository Structure

```
.
├── backend/
│   ├── app.py                     # FastAPI server & REST endpoints
│   ├── face_detector.py           # YuNet face detection & coordinate logic
│   └── gender_model.py            # ResNet-18 + RBF-SVM model loader
├── frontend/
│   ├── index.html                 # Interactive web interface
│   ├── script.js                  # Frontend camera, canvas, and API client
│   └── style.css                  # Responsive UI styling
├── models/
│   ├── face_detection_yunet_2023mar.onnx  # YuNet face detector weights (ONNX)
│   ├── gender_resnet18_best.pth           # Fine-tuned ResNet-18 PyTorch weights
│   └── gender_complete_model.joblib       # Serialized SVM + Scaler + Metadata bundle
├── notebooks/
│   └── SVM_GENDER_TRAINING.ipynb          # End-to-end training & grid search notebook
├── results/
│   ├── dataset_split.csv                  # Stratified train/val/test split index
│   ├── gender_cnn_history.csv             # Training and validation loss curves
│   └── gender_svm_search.csv              # Full SVM hyperparameter search log
├── docs/
│   ├── README.md                          # Theory overview
│   ├── 01_problem_setup.md                # Task formulation & evaluation metrics
│   ├── 02_data_preprocessing.md           # Augmentations & input normalization
│   ├── 03_cnn_backbone.md                 # ResNet-18 feature extractor details
│   ├── 04_svm.md                          # Primal/Dual derivation, KKT conditions, RBF
│   └── 05_detection_deployment.md         # Real-time pipeline, NMS & latency
├── train.py                               # End-to-end data preprocessing & training pipeline
├── README.md                              # Main documentation (this file)
├── requirements.txt                       # Project dependencies
└── .gitignore                             # Ignored files & caches
```

---

## 6. Getting Started & Demo Instructions

### Prerequisites
- Python 3.10 or higher
- Webcam (optional, for live demo)

### Installation
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/prabeshbchettri/FaceClass-CNN-SVM-Based-Facial-Attribute-Classification.git
   cd FaceClass-CNN-SVM-Based-Facial-Attribute-Classification
   ```

2. Switch to the `gender` branch:
   ```bash
   git checkout gender
   ```

3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # On Windows: .venv\Scripts\activate
   ```

4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Web Application
Start the FastAPI server:
```bash
python backend/app.py
```
Open your browser and navigate to:
```
http://localhost:8000
```
- **Image Upload Tab**: Drag-and-drop any photograph to detect faces and view gender predictions with confidence ratings.
- **Live Webcam Tab**: Real-time bounding box detection with color-coded gender overlays (Blue = Male, Pink = Female).

### Running the Data Preprocessing & Training Pipeline
To run a fast live verification of the end-to-end preprocessing, transfer learning, feature extraction, and SVM grid search on synthetic samples:
```bash
python train.py --dry-run
```

To run complete training on the full UTKFace dataset:
```bash
python train.py --data-dir /path/to/UTKFace --epochs 10 --batch-size 32
```

Alternatively, open and run the interactive evaluation notebook:
```bash
jupyter notebook notebooks/SVM_GENDER_TRAINING.ipynb
```

---

## 7. Theoretical Background & Mathematical Derivations

Detailed formal proofs and engineering documentation are provided in the [`docs/`](docs/) directory:
- [**Problem Formulation & Evaluation Metrics**](docs/01_problem_setup.md)
- [**Data Preprocessing & Orientation Augmentation**](docs/02_data_preprocessing.md)
- [**ResNet-18 Feature Extractor Architecture**](docs/03_cnn_backbone.md)
- [**Support Vector Machine: Primal, Dual, KKT & RBF Kernel Proofs**](docs/04_svm.md)
- [**Face Detection, NMS & Real-Time Deployment**](docs/05_detection_deployment.md)

---

## 8. License & Academic Attribution
Developed for academic evaluation and research in Facial Attribute Classification using Deep Learning and Convex Optimization.
