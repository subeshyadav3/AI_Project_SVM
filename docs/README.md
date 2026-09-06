# Mathematical Foundations & System Architecture

This directory provides the formal theoretical derivations, algorithmic formulations, and engineering architecture for the **FaceClass Facial Gender Classification** system.

### Table of Contents

1. [**01. Problem Setup & Metrics**](01_problem_setup.md)
   - Binary gender classification formulation.
   - Stratified dataset partitioning (70% train / 15% val / 15% test).
   - Balanced accuracy, macro F1-score, and per-class recall analysis.
   - Final test performance summary (91.98% accuracy, 91.95% balanced accuracy).

2. [**02. Data & Preprocessing Pipeline**](02_data_preprocessing.md)
   - UTKFace parsing and dataset characteristics.
   - Orientation & tilt robustness augmentation (rotation $\pm 15^\circ$, affine, perspective).
   - Deterministic evaluation transforms and ImageNet channel standardization.

3. [**03. CNN Feature Extractor (ResNet-18)**](03_cnn_backbone.md)
   - Residual learning mathematics and gradient flow guarantees.
   - Transfer learning strategy (freezing `conv1`–`layer3`, fine-tuning `layer4` + `fc`).
   - 512-dimensional bottleneck feature vector extraction via Global Average Pooling.
   - Training optimization (AdamW, label smoothing, gradient clipping, early stopping).

4. [**04. Support Vector Machines (Theory & Derivations)**](04_svm.md)
   - Maximum-margin hyperplane geometry and margin derivation ($\frac{2}{\|\mathbf{w}\|}$).
   - Primal soft-margin formulation with slack variables ($\xi_i$).
   - Lagrangian function and KKT stationarity / complementarity conditions.
   - Dual quadratic programming formulation and support vector sparsity.
   - Radial Basis Function (RBF) kernel mapping into infinite-dimensional Hilbert space.
   - RBF bandwidth parameter $\gamma$ and regularization parameter $C$.
   - Why CNN features + SVM outperforms end-to-end softmax classifiers.

5. [**05. Face Detection & Real-Time Deployment**](05_detection_deployment.md)
   - Real-time OpenCV YuNet ONNX face detection with Non-Maximum Suppression (NMS).
   - Contextual margin padding (15%) for boundary capture.
   - FastAPI asynchronous backend and latency benchmarks (<35 ms on CPU).
   - Interactive web interface for image upload and live webcam evaluation.
