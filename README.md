# FaceClass: Real-Time Facial Gender Classification

A practical, real-time facial gender classification system that combines deep representation learning with classical convex optimization. Built with **PyTorch (ResNet-18)**, an **RBF-Kernel Support Vector Machine (SVM)**, **OpenCV YuNet**, and a lightweight **FastAPI** web interface.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-91.98%25-brightgreen.svg)]()
[![Balanced Accuracy](https://img.shields.io/badge/Balanced_Accuracy-91.95%25-brightgreen.svg)]()

> **Quick Evaluation Notice**:
> All trained model weights, scalers, and detectors are pre-bundled in the [`models/`](models/) directory. **No external downloads, API keys, or GPU required** to run and test the complete system.

---

## Quick Start & Testing Guide (For Evaluators)

Follow these steps to run and test the application on your machine.

### 1. Clone the Repository

```bash
# Using HTTPS (recommended)
git clone https://github.com/subeshyadav3/AI_Project_SVM.git
cd AI_Project_SVM

# Or using SSH
git clone git@github.com:subeshyadav3/AI_Project_SVM.git
cd AI_Project_SVM
```

### 2. Set Up Virtual Environment

> Python 3.10 or 3.11 is recommended.

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

**On Windows (Command Prompt / PowerShell):**
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

*(Optional alternative using [uv](https://github.com/astral-sh/uv) for ultra-fast setup:)*
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r backend/requirements.txt
```

### 3. Quick Terminal Sanity Check (Optional)

To verify in 3 seconds that the models, detector, and PyTorch/SVM pipeline load cleanly:
```bash
python -c "import sys; sys.path.append('backend'); from gender_model import GenderClassifier; from face_detector import FaceDetector; GenderClassifier(); FaceDetector(); print('\n>>> Pipeline & models loaded successfully! <<<')"
```

### 4. Run the Backend & Web Application

Start the server:
```bash
python backend/app.py
```
*(Or alternatively with uvicorn directly: `uvicorn backend.app:app --host 127.0.0.1 --port 8000`)*

### 5. Open & Test in Your Browser

Navigate to:
```
http://localhost:8000
```

You can test the application in three ways:

1. **Upload & Test Images**:
   - Under the **Image Analysis** tab, drag and drop or browse any photograph with one or multiple faces.
   - The app will highlight detected faces with bounding boxes and display predicted gender (`Male` / `Female`) and confidence score.
2. **Live Webcam Test**:
   - Switch to the **Live Webcam** tab and click **Start Camera**.
   - Tests real-time face detection and gender classification at sub-35ms frame latency.
3. **Interactive API Docs (Swagger UI)**:
   - Visit `http://localhost:8000/docs` to test the `/predict` REST endpoint directly with your own images.

---

## What is FaceClass?

Most conventional facial attribute classifiers train a standard deep neural network end-to-end with a softmax output. While functional, softmax decision boundaries often struggle with fine-grained edge cases and require heavy compute during deployment.

**FaceClass** uses a hybrid two-stage architecture:
1. **Deep Feature Extraction (ResNet-18)**: A fine-tuned ResNet-18 convolutional backbone maps any cropped face into a rich 512-dimensional semantic representation vector (via Global Average Pooling).
2. **Maximum-Margin Classification (RBF-SVM)**: An RBF-kernel Support Vector Machine receives the normalized 512-dimensional feature vectors and constructs a mathematically optimal maximum-margin decision boundary. This significantly improves generalization and prevents overfitting.
3. **Real-Time Edge Face Detection (OpenCV YuNet)**: A lightweight ONNX face detector localizes single and multi-face scenes in under 20ms on standard CPUs.

The entire end-to-end pipeline operates at **< 35 ms per image on standard consumer CPUs** without requiring a GPU.

---

## Key Highlights & Contributions

- **Hybrid CNN + SVM Architecture**: Combines the high-level representation power of deep convolutional layers with the theoretical robustness of convex optimization.
- **Fair & Symmetrical Precision**: Class-weighted optimization equalizes recall between male (92.26%) and female (92.48%) faces with virtually zero disparity ($\Delta = 0.22\%$).
- **Lightweight CPU Deployment**: Fast sub-35ms inference latency on regular laptop CPUs.
- **Ready-to-Use UI**: Clean, responsive web frontend with file upload, drag-and-drop, and live webcam feed.
- **Presentation Materials Included**: Slide deck and PDF report are available in [`docs/`](docs/).

---

## Architecture Flow

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

1. **Face Detection**: YuNet locates face coordinates and applies 15% contextual margin padding to preserve jawline, hairline, and facial context.
2. **Feature Extraction**: Face crops are normalized and passed through ResNet-18 to produce a 512-dimensional feature vector.
3. **Classification**: Features are standardized with `StandardScaler` and classified by the RBF Support Vector Machine.

---

## Quantitative Results

Evaluated on the held-out test split of the **UTKFace** benchmark dataset:

| Metric | Score | Detail |
|---|---|---|
| **Overall Accuracy** | **91.98%** | Total test samples correctly classified |
| **Balanced Accuracy** | **91.95%** | Mean recall across both male and female classes |
| **Macro F1-Score** | **0.9196** | Harmonic mean of precision and recall |
| **Male Recall** | **92.26%** | Correctly predicted male subjects |
| **Female Recall** | **92.48%** | Correctly predicted female subjects |
| **Recall Disparity ($\Delta$)** | **0.22%** | Symmetrical performance eliminating gender bias |

### Optimal Hyperparameters:
- **CNN Backbone**: ResNet-18 fine-tuned on `layer4` + `fc` (AdamW, lr = $10^{-4}$, Dropout = 0.45)
- **Feature Dimension**: 512 (Global Average Pooling)
- **SVM Kernel**: Radial Basis Function (RBF)
- **Regularization ($C$)**: 0.01
- **Gamma ($\gamma$)**: `'scale'` ($1 / (d \cdot \sigma^2)$)
- **Class Weights**: `'balanced'`

---

## Project Presentation & Documents

The presentation files are available in [`docs/`](docs/):
- **Presentation Slide Deck (PPTX)**: [`docs/AIProject_SVM.pptx`](docs/AIProject_SVM.pptx)
- **Presentation Document (PDF)**: [`docs/AIProject_SVM.pdf`](docs/AIProject_SVM.pdf)

---

## Retraining from Scratch (Optional)

If you wish to re-train the models or re-run the SVM grid search:

1. **Dry-run verification** (tests preprocessing and grid search pipeline on synthetic data):
   ```bash
   python train.py --dry-run
   ```

2. **Full training on UTKFace dataset**:
   ```bash
   # First install full training dependencies
   pip install -r requirements.txt
   python train.py --data-dir /path/to/UTKFace --epochs 10 --batch-size 32
   ```

3. **Interactive Jupyter Notebook**:
   ```bash
   jupyter notebook notebooks/SVM_GENDER_TRAINING.ipynb
   ```

---

## Repository Structure

```
.
├── backend/
│   ├── app.py                     # FastAPI web server and REST endpoints
│   ├── face_detector.py           # OpenCV YuNet face detection & coordinate logic
│   ├── gender_model.py            # ResNet-18 feature extraction & SVM classifier
│   └── requirements.txt           # Minimal backend dependencies for deployment
├── frontend/
│   ├── index.html                 # Clean, responsive web UI
│   ├── script.js                  # Frontend webcam stream, canvas & API client
│   └── style.css                  # UI styling and visual themes
├── models/
│   ├── face_detection_yunet_2023mar.onnx  # Pretrained YuNet ONNX face detector
│   ├── gender_resnet18_best.pth           # Fine-tuned ResNet-18 feature extractor weights
│   └── gender_complete_model.joblib       # Serialized SVM model + StandardScaler bundle
├── notebooks/
│   └── SVM_GENDER_TRAINING.ipynb          # End-to-end training and SVM grid search notebook
├── results/
│   ├── dataset_split.csv                  # Stratified train/val/test splits
│   ├── gender_cnn_history.csv             # Training and validation loss curves
│   └── gender_svm_search.csv              # SVM hyperparameter search log
├── docs/
│   ├── AIProject_SVM.pdf                  # Presentation document (PDF)
│   └── AIProject_SVM.pptx                 # Presentation slide deck (PPTX)
├── train.py                               # Training, feature extraction & grid search pipeline
├── requirements.txt                       # Full project dependencies (training + backend)
└── README.md                              # Main documentation & testing guide
```

---

## License & Academic Attribution

Developed for academic evaluation and research in Facial Attribute Classification using Deep Learning and Convex Optimization.
