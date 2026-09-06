# 4. Support Vector Machines: Mathematical Derivations & Theory

The core decision engine of this project is a **Soft-Margin Support Vector Machine with a Radial Basis Function (RBF) Kernel**, trained on standardized 512-D deep features extracted from the fine-tuned ResNet-18 backbone.

This section provides the rigorous mathematical formulation and derivations underlying the classifier.

---

## 4.1 Maximum-Margin Hyperplane Geometry

Let the training dataset of extracted embeddings be:

$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}, \quad \mathbf{x}_i \in \mathbb{R}^{d} \; (d=512), \quad y_i \in \{-1, +1\}$$

where $y_i = -1$ corresponds to **Male** and $y_i = +1$ corresponds to **Female**.

A separating hyperplane is defined by a normal weight vector $\mathbf{w} \in \mathbb{R}^d$ and a scalar bias $b \in \mathbb{R}$:

$$\mathcal{H} = \{\mathbf{x} \in \mathbb{R}^d : \mathbf{w}^\top \mathbf{x} + b = 0\}$$

The orthogonal geometric distance from an arbitrary point $\mathbf{x}_i$ to the hyperplane is:

$$d_i = \frac{y_i(\mathbf{w}^\top \mathbf{x}_i + b)}{\|\mathbf{w}\|_2}$$

By scaling $(\mathbf{w}, b)$ such that the closest training points (the **support vectors**) satisfy $| \mathbf{w}^\top \mathbf{x}_i + b | = 1$, the canonical bounding hyperplanes become:

$$\mathcal{H}_+ : \mathbf{w}^\top \mathbf{x} + b = +1, \qquad \mathcal{H}_- : \mathbf{w}^\top \mathbf{x} + b = -1$$

The geometric margin $\gamma_{\text{margin}}$—the perpendicular distance between $\mathcal{H}_+$ and $\mathcal{H}_-$—is:

$$\gamma_{\text{margin}} = \frac{1 - (-1)}{\|\mathbf{w}\|_2} = \frac{2}{\|\mathbf{w}\|_2}$$

Maximizing the margin $\frac{2}{\|\mathbf{w}\|_2}$ is mathematically equivalent to minimizing $\frac{1}{2}\|\mathbf{w}\|_2^2$.

---

## 4.2 Non-Separable Case: Soft-Margin Primal Formulation

Because real-world facial representations cannot be cleanly separated by a linear boundary, we introduce non-negative **slack variables** $\xi_i \ge 0$ to permit controlled margin violations:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad \frac{1}{2}\|\mathbf{w}\|_2^2 + C \sum_{i=1}^{n} \xi_i$$

$$\text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0, \quad \forall i \in \{1, \dots, n\}$$

### Interpretation of Slack Variables:
- $\xi_i = 0$: Point lies strictly on the correct side of the margin boundary.
- $0 < \xi_i < 1$: Point is correctly classified but falls within the margin corridor.
- $\xi_i = 1$: Point lies exactly on the decision boundary.
- $\xi_i > 1$: Point is misclassified.

### Role of the Penalty Parameter $C$:
$C > 0$ controls the fundamental **bias-variance tradeoff**:
- **Large $C$**: Heavy penalty for margin violations; produces a narrower, tighter margin (low bias, higher variance / risk of overfitting).
- **Small $C$**: Tolerates more boundary violations; produces a wider, smoother margin (higher bias, lower variance / robust generalization).

In our empirical validation sweep, $C = 0.01$ with `class_weight='balanced'` yielded optimal generalization, preventing overfitting on noisy facial edge-cases.

---

## 4.3 Lagrangian Dual Derivation

To solve the constrained convex optimization problem and enable kernelization, we construct the generalized Lagrangian. Let $\boldsymbol{\alpha} = (\alpha_1, \dots, \alpha_n)^\top \ge \mathbf{0}$ be Lagrange multipliers for the margin constraints, and $\boldsymbol{\mu} = (\mu_1, \dots, \mu_n)^\top \ge \mathbf{0}$ be multipliers for the non-negativity of slack variables:

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu}) = \frac{1}{2}\|\mathbf{w}\|_2^2 + C \sum_{i=1}^{n} \xi_i - \sum_{i=1}^{n} \alpha_i \big[ y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 + \xi_i \big] - \sum_{i=1}^{n} \mu_i \xi_i$$

### First-Order Karush-Kuhn-Tucker (KKT) Stationarity Conditions:
Differentiating with respect to the primal variables and setting gradients to zero:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{w} - \sum_{i=1}^{n} \alpha_i y_i \mathbf{x}_i = \mathbf{0} \implies \mathbf{w} = \sum_{i=1}^{n} \alpha_i y_i \mathbf{x}_i$$

$$\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^{n} \alpha_i y_i = 0 \implies \sum_{i=1}^{n} \alpha_i y_i = 0$$

$$\frac{\partial \mathcal{L}}{\partial \xi_i} = C - \alpha_i - \mu_i = 0 \implies C = \alpha_i + \mu_i$$

Since multipliers must be non-negative ($\mu_i \ge 0$), the condition $C - \alpha_i = \mu_i \ge 0$ yields the **box constraint**:

$$0 \le \alpha_i \le C, \quad \forall i \in \{1, \dots, n\}$$

### Substituting into the Lagrangian (The Dual Problem):
Substituting $\mathbf{w} = \sum_{i} \alpha_i y_i \mathbf{x}_i$ and $\sum_i \alpha_i y_i = 0$ back into $\mathcal{L}$ yields the Wolfe Dual formulation:

$$\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j (\mathbf{x}_i^\top \mathbf{x}_j)$$

$$\text{subject to} \quad 0 \le \alpha_i \le C, \quad \sum_{i=1}^{n} \alpha_i y_i = 0$$

This is a **concave quadratic programming (QP) problem** with linear constraints. It guarantees a **unique global optimum** $\boldsymbol{\alpha}^*$, completely eliminating local minima traps common in deep neural networks.

---

## 4.4 KKT Complementary Slackness & Support Vector Sparsity

At the optimal solution $(\mathbf{w}^*, b^*, \boldsymbol{\xi}^*, \boldsymbol{\alpha}^*, \boldsymbol{\mu}^*)$, the KKT complementary slackness conditions must hold:

$$\alpha_i^* \big[ y_i(\mathbf{w}^{*\top} \mathbf{x}_i + b^*) - 1 + \xi_i^* \big] = 0$$

$$\mu_i^* \xi_i^* = (C - \alpha_i^*) \xi_i^* = 0$$

This partitions the training set into three distinct categories:

| Multiplier Value | Geometric Location | Physical Meaning |
|---|---|---|
| **$\alpha_i = 0$** | $y_i(\mathbf{w}^\top \mathbf{x}_i + b) > 1$ | Non-support vector. Correctly classified, outside the margin. Does not affect the decision boundary. |
| **$0 < \alpha_i < C$** | $y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1, \; \xi_i = 0$ | **Free Support Vector**. Lies exactly on the margin boundary. Used to solve for the optimal bias $b^*$. |
| **$\alpha_i = C$** | $\xi_i > 0$ | **Bounded Support Vector**. Margin violator (either inside margin or misclassified). |

Because most training samples have $\alpha_i = 0$, the decision surface depends **solely on the sparse set of support vectors** ($\mathcal{SV} = \{i : \alpha_i > 0\}$):

$$f(\mathbf{x}) = \sum_{i \in \mathcal{SV}} \alpha_i y_i (\mathbf{x}_i^\top \mathbf{x}) + b$$

---

## 4.5 Kernelization & The Radial Basis Function (RBF) Kernel

In non-linear feature spaces, mapping inputs into a higher-dimensional Hilbert space $\Phi(\mathbf{x}) \in \mathcal{H}$ allows linear separation:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \langle \Phi(\mathbf{x}_i), \Phi(\mathbf{x}_j) \rangle_{\mathcal{H}}$$

By Mercer's Theorem, if $K$ is positive semi-definite, we can compute high-dimensional inner products without ever explicitly computing $\Phi(\mathbf{x})$ (the "Kernel Trick").

We employ the **Gaussian Radial Basis Function (RBF) Kernel**:

$$K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|_2^2\right), \quad \gamma > 0$$

### Properties of the RBF Kernel:
1. **Infinite-Dimensional Feature Space**: The Taylor series expansion of $\exp(z)$ contains infinitely many polynomial terms, enabling the RBF kernel to model arbitrary non-linear decision manifolds.
2. **Local Influence**: $K(\mathbf{x}, \mathbf{x}') \to 1$ when $\|\mathbf{x} - \mathbf{x}'\| \to 0$, and decays exponentially to $0$ as points move further apart.
3. **Bandwidth Parameter $\gamma$**:
   - Small $\gamma$: Broad Gaussian curves; decision boundary is smooth and gradual.
   - Large $\gamma$: Tight Gaussian peaks; decision boundary is localized and risks overfitting training points.
   - We utilize $\gamma = \text{'scale'} = \frac{1}{d \cdot \operatorname{Var}(X)}$, which dynamically scales with feature dimensionality ($d=512$) and dataset variance.

---

## 4.6 Feature Normalization via `StandardScaler`

The RBF kernel is strictly a function of the **squared Euclidean distance** $\|\mathbf{x} - \mathbf{x}'\|_2^2 = \sum_{j=1}^{d} (x_j - x_j')^2$.

If unstandardized, high-magnitude feature dimensions would dominate distance calculations, while subtle dimensions would be suppressed. We fit a `StandardScaler` strictly on the training set:

$$\tilde{z}_j = \frac{z_j - \mu_j}{\sigma_j}, \quad \mu_j = \frac{1}{n_{\text{train}}}\sum_{i=1}^{n_{\text{train}}} z_{ij}, \quad \sigma_j = \sqrt{\frac{1}{n_{\text{train}}}\sum_{i=1}^{n_{\text{train}}} (z_{ij} - \mu_j)^2}$$

The computed mean and variance vectors are serialized inside `models/gender_complete_model.joblib` and applied identically during inference to prevent data leakage.

---

## 4.7 Class-Weight Balancing

To prevent bias toward the majority demographic, we scale the penalty parameter $C$ inversely proportional to class frequencies:

$$C_k = C \cdot w_k, \qquad w_k = \frac{N}{2 \cdot N_k}$$

where $N$ is the total number of training samples, and $N_k$ is the count for class $k$. Margin violations on the minority class are penalized more heavily, enforcing symmetric margins for both genders.

---

## 4.8 Decision Function and Probability Calibration

The continuous decision margin for an input face embedding $\tilde{\mathbf{z}}$ is:

$$f(\tilde{\mathbf{z}}) = \sum_{i \in \mathcal{SV}} \alpha_i y_i \exp\left(-\gamma \|\tilde{\mathbf{x}}_i - \tilde{\mathbf{z}}\|_2^2\right) + b$$

The predicted class is:

$$\hat{y} = \begin{cases} \text{Female} & \text{if } f(\tilde{\mathbf{z}}) > \tau \\ \text{Male} & \text{if } f(\tilde{\mathbf{z}}) \le \tau \end{cases}$$

where $\tau = 0.0$ is the standard neutral boundary.

To provide user-friendly confidence percentages in the web interface, the signed margin distance is mapped via a calibrated sigmoid transformation:

$$P(\text{Female} \mid \mathbf{x}) = \frac{1}{1 + \exp(-f(\tilde{\mathbf{z}}))}$$

---

## 4.9 Why CNN Feature Extractor + SVM vs. End-to-End Softmax

1. **Global Convexity**: The SVM dual formulation is strictly convex. Once CNN features are extracted, finding the optimal boundary is deterministic and globally optimal, without random seed variance.
2. **Margin Maximization**: Standard softmax cross-entropy minimizes log-loss but does not explicitly maximize the geometric margin between classes. SVM directly optimizes separation distance, providing superior generalization on unseen poses.
3. **Decoupled Architecture**: Allows independent optimization of representation learning (CNN feature tuning) and decision surface geometry (RBF-SVM).
