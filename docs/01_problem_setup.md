# 1. Problem Setup: Facial Gender Classification

## 1.1 Task Definition

Facial gender classification is formulated as a supervised binary classification problem:

$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}, \quad \mathbf{x}_i \in \mathbb{R}^{224 \times 224 \times 3}, \quad y_i \in \{-1, +1\}$$

where:
- $\mathbf{x}_i$ is an RGB face image cropped and normalized from in-the-wild photography.
- $y_i = -1$ denotes **Male** (index 0).
- $y_i = +1$ denotes **Female** (index 1).

The model maps face image $\mathbf{x}$ to class prediction $\hat{y}$ through a hybrid deep feature extraction and maximum-margin kernel classification pipeline:

$$\mathbf{x} \xrightarrow{\text{ResNet-18 } \phi(\cdot)} \mathbf{z} \in \mathbb{R}^{512}
  \xrightarrow{\text{StandardScaler}} \tilde{\mathbf{z}} \in \mathbb{R}^{512}
  \xrightarrow{\text{RBF-SVM } f(\cdot)} \hat{y} = \operatorname{sign}(f(\tilde{\mathbf{z}})).$$

## 1.2 Data Splitting Strategy

The dataset is partitioned into three disjoint sets:
- **Training Set (70%)**: Used to fine-tune the ResNet-18 feature extractor and fit the candidate SVM classifiers.
- **Validation Set (15%)**: Used for hyperparameter tuning ($C, \gamma$) via grid search and early stopping.
- **Test Set (15%)**: Held-out set evaluated only once to report final performance without data leakage.

Stratification is strictly applied so both training and evaluation subsets preserve identical gender and demographic distributions.

## 1.3 Evaluation Metrics

Given $N$ test samples with true labels $y_i \in \{-1, +1\}$ and predictions $\hat{y}_i$:

### Accuracy
The overall proportion of correct predictions:

$$\mathrm{Accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\hat{y}_i = y_i]$$

While standard, overall accuracy alone can mask systemic error asymmetry (e.g., higher error on one gender).

### Per-Class Recall & Precision
For each gender $k \in \{\text{Male}, \text{Female}\}$:

$$\mathrm{Recall}_k = \frac{\mathrm{TP}_k}{\mathrm{TP}_k + \mathrm{FN}_k}, \qquad \mathrm{Precision}_k = \frac{\mathrm{TP}_k}{\mathrm{TP}_k + \mathrm{FP}_k}$$

### Balanced Accuracy (Primary Selection Criterion)
The unweighted mean of recall across both genders:

$$\mathrm{Balanced\text{-}Accuracy} = \frac{1}{2}\left(\mathrm{Recall}_{\text{Male}} + \mathrm{Recall}_{\text{Female}}\right)$$

Balanced accuracy is the objective criterion optimized during SVM grid search (`balanced_accuracy_score`). This explicitly penalizes any model that achieves high overall accuracy by disproportionately favoring one gender.

### Macro F1-Score
The harmonic mean of precision and recall averaged across both classes:

$$\mathrm{Macro\text{-}F1} = \frac{F1_{\text{Male}} + F1_{\text{Female}}}{2}$$

## 1.4 Test Evaluation Results

On the held-out test evaluation set, the hybrid ResNet-18 + RBF-SVM achieves:

| Metric | Score | Note |
|---|---|---|
| **Test Accuracy** | **91.98%** | High overall discrimination |
| **Balanced Accuracy** | **91.95%** | Symmetrical performance |
| **Macro F1-Score** | **0.9196** | Harmonized precision & recall |
| **Male Recall** | **92.26%** | Accurate male identification |
| **Female Recall** | **92.48%** | Accurate female identification |
| **Recall Disparity ($\Delta$)** | **0.22%** | Minimal Male $\leftrightarrow$ Female error gap |
