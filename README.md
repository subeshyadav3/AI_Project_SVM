# Gender Classification by Facial Attribute Using SVM

> An end-to-end facial attribute classification system combining transfer-learned ResNet-18 deep feature extraction with a maximum-margin Support Vector Machine classifier on the UTKFace benchmark.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Test Accuracy](https://img.shields.io/badge/Test_Accuracy-91.98%25-brightgreen.svg)]()
[![Validation Accuracy](https://img.shields.io/badge/Val_Accuracy-91.51%25-brightgreen.svg)]()

---

## Authors & Project Team

| Name | Roll Number |
|---|---|
| **Shreeyut Thapa** | `080BCT080` |
| **Prabesh BC** | `080BCT054` |
| **Subesh Yadav** | `080BCT084` |

---

> **Notice for Evaluators**:
> All trained model weights, scalers, and face detectors are pre-bundled inside [`models/`](models/). **No external downloads, API keys, or GPU are required** to test and run the full application.

---

## Quick Start & Testing Guide (For Evaluators)

Follow these steps to test the project locally:

### 1. Clone the Repository

```bash
# Using HTTPS (recommended)
git clone https://github.com/subeshyadav3/AI_Project_SVM.git
cd AI_Project_SVM

# Or using SSH
git clone git@github.com:subeshyadav3/AI_Project_SVM.git
cd AI_Project_SVM
```

### 2. Set Up Virtual Environment & Dependencies

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

*(Alternative using [uv](https://github.com/astral-sh/uv) for fast install:)*
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r backend/requirements.txt
```

### 3. Quick Terminal Sanity Check (3 Seconds)

Verify that the models, detector, and PyTorch/SVM pipeline load cleanly:
```bash
python -c "import sys; sys.path.append('backend'); from gender_model import GenderClassifier; from face_detector import FaceDetector; GenderClassifier(); FaceDetector(); print('\n>>> Pipeline & models loaded successfully! <<<')"
```

### 4. Run the Web Application

```bash
python backend/app.py
```
*(Or via uvicorn directly: `uvicorn backend.app:app --host 127.0.0.1 --port 8000`)*

### 5. Open in Browser

Open:
```
http://localhost:8000
```

- **Image Upload Tab**: Drag-and-drop or select any picture containing faces to view bounding boxes, predicted gender (`Male` / `Female`), and confidence ratings.
- **Live Webcam Tab**: Click **Start Camera** to test real-time detection on webcam video (<35ms latency).
- **Interactive API Docs**: Visit `http://localhost:8000/docs` to test the `/predict` REST endpoint directly via Swagger UI.

---

## Project Overview & Motivation

Automated facial attribute classification faces significant real-world challenges:
- **Morphological Variance**: Non-linear diversity driven by age, ethnicity, illumination, and head orientation.
- **Subtle Dimorphism**: Secondary sexual characteristics (jawline contour, brow prominence, cheekbone geometry) require sensitive spatial representations.
- **Classical Limitations**: Handcrafted descriptors (HOG, LBP, Haar cascades) degrade under unconstrained pose, lighting, and expressions.

### The Proposed Hybrid Solution
Rather than relying purely on an end-to-end neural network with a standard softmax loss, this project implements a **decoupled hybrid architecture**:
1. **Deep Feature Extraction (ResNet-18)**: Transfer-learned ResNet-18 transforms 224×224 RGB faces into compact 512-dimensional bottleneck embeddings.
2. **Maximum-Margin Boundary (RBF-SVM)**: An RBF-kernel Support Vector Machine finds a globally optimal separating hyperplane, avoiding local minima inherent to pure cross-entropy backpropagation.
3. **Decoupled Modularity**: Allows independent feature-space normalization, disciplined regularization, and low-latency (<35ms) CPU inference.

---

## Architecture Pipeline

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

### Inference Flow:
- **Phase 1 (Ingest & Decode)**: User image uploaded via multipart form $\rightarrow$ FastAPI async endpoint (`POST /predict`) $\rightarrow$ OpenCV image decoding.
- **Phase 2 (Face Localization)**: OpenCV YuNet ONNX face detector (score $\ge 0.60$, min 40px, Haar fallback) $\rightarrow$ crop with +15% border margin $\rightarrow$ Resize 256 $\rightarrow$ CenterCrop 224.
- **Phase 3 (Feature Backbone)**: Forward pass through fine-tuned ResNet-18 (`fc` $\rightarrow$ Identity) under `torch.no_grad()` $\rightarrow$ spatially pooled 512-D embedding vector.
- **Phase 4 (Classification & Output)**: `StandardScaler` normalization $\rightarrow$ RBF-SVM signed margin ($w^T x + b$) $\rightarrow$ sigmoid confidence clipped $[0.50, 0.99]$ $\rightarrow$ JSON response with annotated bounding boxes.

---

## ResNet-18 Backbone & Layer Freezing Strategy

Residual skip connections $y = \mathcal{F}(x) + x$ resolve vanishing gradients. Freezing early layers preserves general low/mid-level visual filters while unfreezing Layer 4 tailors high-level representations to facial dimorphism:

| Stage | Output Dimension | Visual Representation | Status |
|---|---|---|---|
| **Input** | $3 \times 224 \times 224$ RGB | Face crop (Resize 256 $\rightarrow$ CenterCrop 224) | — |
| **Conv1 + MaxPool** | $64 \text{ ch}, 56 \times 56$ | Edges, gradients, low-level textures | **Frozen** |
| **Layers 1–3** | $256 \text{ ch}, 14 \times 14$ | Mid-level contours, eye sockets, nose bridge | **Frozen** |
| **Layer 4** | $512 \text{ ch}, 7 \times 7$ | High-level dimorphism: jaw, brow, cheekbones | **Fine-Tuned** |
| **AdaptiveAvgPool2d (GAP)** | $1 \times 512$ vector | Spatially invariant embedding $\rightarrow$ Scaler & SVM | **Extracted** |

---

## Theoretical Foundation: SVM & RBF Kernel

### 1. Structural Risk Minimization
A Support Vector Machine constructs an optimal separating hyperplane by maximizing the geometric margin $\gamma = \frac{2}{\|w\|}$:
$$\min_{w, b} \frac{1}{2}\|w\|^2 \quad \text{subject to } y_i(w^T x_i + b) \ge 1, \quad \forall i \in \{1, \dots, N\}$$

### 2. Soft-Margin Optimization & Slack Variables
To handle demographic overlap and ambiguous facial features:
$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t. } y_i(w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
- **Role of Slack ($\xi_i$)**: Quantifies margin violations.
- **Regularization ($C$)**: Controls margin width vs. violation penalty.
- **Why $C = 0.01$ won**: Grid search revealed that larger $C$ values forced the model to memorize noisy demographic boundaries (infants and elderly). $C = 0.01$ established a smooth, highly regularized margin with peak generalization.

### 3. Non-Linear Dual Formulation & RBF Kernel
$$\max_{\alpha} \sum_{i} \alpha_i - \frac{1}{2}\sum_{i, j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$
where $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$ and $\gamma = \text{'scale'} = \frac{1}{d \cdot \operatorname{Var}(X)}$ adapts to the 512 feature dimensions.

---

## Dataset & Stratified Partitioning

Evaluated on the **UTKFace** benchmark corpus:
- **Total Samples**: 24,102 face images
- **Demographic Split**: 52.2% Male (12,581) / 47.8% Female (11,521)
- **Filename Parsing**: `[age]_[gender]_[race]_[date&time].jpg` (Gender token: `0` = Male, `1` = Female)

| Partition | Share | Count | Purpose |
|---|---|---|---|
| **Training Set** | 70.0% | 16,871 faces | Fine-tuning ResNet-18 Layer 4 + FC; initial SVM fit |
| **Validation Set** | 15.0% | 3,615 faces | Grid search hyperparameter optimization (36 parameter pairs) |
| **Test Set (Holdout)** | 15.0% | 3,616 faces | Untouched final benchmark evaluation |
| **Merged Retraining Set** | 85.0% | 20,486 faces | Train + Val merged fit before final holdout testing |

---

## Hyperparameter Grid Search & Results

Exhaustive search across 36 $(C, \gamma)$ combinations:
- **Search Space**: $C \in \{0.01, 0.1, 1, 10, 100, 1000\}$ and $\gamma \in \{\text{'scale'}, \text{'auto'}, 0.001, 0.01, 0.1, 1\}$
- **Optimal Hyperparameters**: $C = 0.01$, $\gamma = \text{'scale'}$
- **Best Validation Accuracy**: **91.51%**

### Final Benchmark on Untouched Holdout Test Set (3,616 faces):

| Metric | Result |
|---|---|
| **Overall Test Accuracy** | **91.98%** (3,225 / 3,616 correct classifications) |
| **Total Test Errors** | Only 285 total misclassifications across test set |

### Performance by Class:

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Male (0)** | 91.5% | 93.0% | 92.2% | 1,859 |
| **Female (1)** | 92.5% | 90.9% | 91.7% | 1,698 |
| **Overall Model** | **92.0%** | **92.0%** | **92.0%** | **Acc: 91.98%** |

- **Balanced Disparity**: Less than 2% recall difference between genders ($93.0\%$ vs $90.9\%$).
- **Strong Margin Separation**: Verified by balanced F1-scores ($92.2\%$ Male vs. $91.7\%$ Female).

---

## Project Presentation Materials

Full presentation files from the project defense are available directly in the repository:
- **Presentation Slide Deck (PowerPoint)**: [`docs/AIProject_SVM.pptx`](docs/AIProject_SVM.pptx)
- **Presentation Document (PDF)**: [`docs/AIProject_SVM.pdf`](docs/AIProject_SVM.pdf)

---

## Repository Structure

```
.
├── backend/
│   ├── app.py                     # FastAPI server & inference endpoints
│   ├── face_detector.py           # OpenCV YuNet detector & coordinate logic
│   ├── gender_model.py            # ResNet-18 GAP extractor + RBF-SVM pipeline
│   └── requirements.txt           # Minimal backend dependencies
├── frontend/
│   ├── index.html                 # Responsive web interface
│   ├── script.js                  # Camera stream, canvas overlay & API caller
│   └── style.css                  # UI theme and layout styling
├── models/
│   ├── face_detection_yunet_2023mar.onnx  # Pretrained YuNet ONNX face detector
│   ├── gender_resnet18_best.pth           # Fine-tuned ResNet-18 PyTorch weights
│   └── gender_complete_model.joblib       # Bundled SVM model + StandardScaler + metadata
├── notebooks/
│   └── SVM_GENDER_TRAINING.ipynb          # End-to-end training & grid search notebook
├── results/
│   ├── dataset_split.csv                  # Stratified train/val/test splits
│   ├── gender_cnn_history.csv             # Training and validation loss curves
│   └── gender_svm_search.csv              # SVM hyperparameter search log
├── docs/
│   ├── AIProject_SVM.pdf                  # Presentation document (PDF)
│   └── AIProject_SVM.pptx                 # Presentation slide deck (PPTX)
├── train.py                               # Complete training & grid search script
├── requirements.txt                       # Full project dependencies (training + backend)
└── README.md                              # Main documentation & testing guide
```

---

## Retraining Pipeline (Optional)

To re-run training or reproduce the grid search:

1. **Dry-Run Sanity Check** (verifies training code on synthetic samples):
   ```bash
   python train.py --dry-run
   ```

2. **Full Training on UTKFace**:
   ```bash
   pip install -r requirements.txt
   python train.py --data-dir /path/to/UTKFace --epochs 10 --batch-size 32
   ```

3. **Interactive Jupyter Notebook**:
   ```bash
   jupyter notebook notebooks/SVM_GENDER_TRAINING.ipynb
   ```

---

## Academic Attribution & License

Developed as an academic research project in Facial Attribute Classification utilizing Deep Representation Learning and Support Vector Machines.
