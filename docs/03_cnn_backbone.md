# 3. CNN Feature Extractor: ResNet-18 Backbone

## 3.1 Residual Learning Architecture

Deep convolutional networks suffer from the degradation problem: as depth increases, accuracy saturates and degrades due to vanishing gradients. ResNet addresses this by formulating layers as learning residual functions with reference to layer inputs:

$$\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}; \mathcal{W}) + \mathbf{x}$$

where $\mathbf{x}$ is the input vector, $\mathcal{H}(\mathbf{x})$ is the underlying mapping, and $\mathcal{F}(\mathbf{x}; \mathcal{W})$ is the residual mapping to be learned.

Gradient backpropagation benefits from the direct additive identity shortcut:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathcal{H}} \left(\frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I}\right)$$

Even if the derivative $\frac{\partial \mathcal{F}}{\partial \mathbf{x}}$ approaches zero, the identity matrix $\mathbf{I}$ ensures continuous, unhindered gradient flow back to the initial layers.

## 3.2 Transfer Learning & Stage-Wise Fine-Tuning

Training a deep CNN from scratch on facial data is prone to severe overfitting. Therefore, we initialize ResNet-18 with ImageNet-pretrained weights (`weights=ResNet18_Weights.DEFAULT`):

1. **Frozen Low-Level Layers (`conv1` through `layer3`)**: Early filters capture generic visual primitives (Gabor-like edges, textures, corner junctions) that transfer universally across vision domains. Freezing them preserves feature fidelity and cuts training time.
2. **Fine-Tuned High-Level Layers (`layer4` + `fc`)**: The final convolutional stage (`layer4`, containing 512 channels) is unfrozen to learn face-specific semantic patterns (jawline curvature, orbital rim geometry, brow structure).
3. **Regularized Classifier Head**:
   ```python
   self.backbone.fc = nn.Sequential(
       nn.Dropout(p=0.45),
       nn.Linear(512, 2)
   )
   ```
   Strong dropout ($p=0.45$) prevents co-adaptation of neurons, acting as an ensemble regularizer.

## 3.3 512-Dimensional Deep Embedding Extraction

After fine-tuning, the temporary classification layer `fc` is bypassed. The model operates as a high-level feature extractor:

$$\mathbf{x} \xrightarrow{\text{Conv1} \to \text{MaxPool} \to \text{Layers 1--4}} \mathbf{F} \in \mathbb{R}^{512 \times 7 \times 7} \xrightarrow{\text{Global Avg Pooling}} \mathbf{z} \in \mathbb{R}^{512}$$

Global Average Pooling (GAP) aggregates spatial activations across each $7 \times 7$ feature map into a single scalar, conferring spatial translation invariance and producing a compact, fixed 512-D embedding $\mathbf{z}$.

## 3.4 Optimization & Overfitting Countermeasures

As defined in `notebooks/SVM_GENDER_TRAINING.ipynb`:

- **Loss Function**: Cross-entropy with **Label Smoothing** ($\varepsilon = 0.05$):
  $$q_k = (1 - \varepsilon)\mathbf{1}[k = y] + \frac{\varepsilon}{K}$$
  This prevents the softmax outputs from growing unbounded and producing overconfident, fragile decision boundaries.
- **Optimizer**: **AdamW** with weight decay $\lambda = 10^{-4}$ to decouple $L_2$ weight regularization from gradient updates.
- **Differential Learning Rates**:
  - Backbone `layer4`: $\eta_1 = 1 \times 10^{-4}$ (cautious updates to preserve pretrained representations).
  - Head `fc`: $\eta_2 = 5 \times 10^{-4}$ (faster convergence for the newly initialized linear layer).
- **Learning Rate Schedule**: `ReduceLROnPlateau(mode='max', factor=0.5, patience=1)` halving the learning rate when validation balanced accuracy stalls.
- **Gradient Clipping**: Norm capped at $5.0$ to prevent gradient instability during backpropagation.
- **Early Stopping**: Monitored on validation balanced accuracy with best checkpoint restoration (`models/gender_resnet18_best.pth`).
