# 2. Data & Preprocessing Pipeline

## 2.1 UTKFace Dataset & Label Extraction

The model is trained on the large-scale **UTKFace** benchmark dataset containing over 20,000 cropped face images with variations in pose, facial expression, illumination, and ethnicity.

Filenames follow the standardized format:
```
[age]_[gender]_[race]_[date&time].jpg
```
Example: `25_0_2_20170116174525116.jpg`
- `age`: 25 years old
- `gender`: `0` = Male, `1` = Female
- `race`: `2` = Asian

Labels are extracted programmatically. Any corrupted images or invalid label codes are safely filtered during parsing.

## 2.2 Orientation & Robustness Augmentation

Real-world deployment scenarios (webcams, smartphone cameras, candid photos) often involve non-frontal head poses and camera tilt. Standard CNN models trained only on perfectly upright faces exhibit severe performance degradation when faces are slightly tilted.

To guarantee tilt and pose invariance, the training pipeline employs an **orientation-aware data augmentation policy**:

1. **Random Rotation ($\pm 15^\circ$)**: Teaches rotational invariance for tilted heads.
2. **Random Affine Transformations**: Translation ($\pm 5\%$) and shear ($\pm 8^\circ$) to simulate varied camera angles.
3. **Random Perspective Jitter ($p=0.20$)**: Simulates 3D out-of-plane perspective shifts.
4. **Random Horizontal Flip ($p=0.50$)**: Leverages natural bilateral facial symmetry.
5. **Color Jitter (Brightness 0.15, Contrast 0.15, Saturation 0.10)**: Simulates diverse lighting environments (shadows, direct sunlight, fluorescent room lighting).
6. **Gaussian Blur & Random Erasing**: Prevents the network from over-indexing on localized artifacts (e.g., specific earrings or background cues).

## 2.3 Input Resolution & Preprocessing Pipeline

### Training Transform Pipeline
```python
transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Evaluation & Inference Transform Pipeline
Inference must be strictly deterministic without stochastic noise:
```python
transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## 2.4 ImageNet Normalization Rationale

Raw RGB pixel intensities reside in $[0, 255]$. Transforming to tensors maps values to $[0.0, 1.0]$. Applying channel-wise standardization:

$$\tilde{x}_c = \frac{x_c - \mu_c}{\sigma_c}, \quad \mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

centers the inputs around zero with unit variance. This conditioning is essential because:
1. ResNet-18 weights pretrained on ImageNet operate in this standardized domain.
2. It prevents vanishing or exploding gradients during backpropagation.
3. It guarantees that the 512-D feature space extracted by the CNN maintains a consistent geometric scale across all samples.
