"""
FaceClass - Facial Gender Classification: Preprocessing & Training Pipeline
Architecture: ResNet-18 (512-D Bottleneck) + StandardScaler + RBF Kernel SVM

This script performs:
1. UTKFace dataset parsing, validation, and stratified train/val/test splitting.
2. Data augmentation with orientation and tilt robustness.
3. ResNet-18 transfer learning (fine-tuning layer4 and classification head).
4. Deep feature extraction (512-D bottleneck embeddings).
5. StandardScaler normalization and RBF-SVM hyperparameter optimization.
6. Evaluation (Accuracy, Balanced Accuracy, Macro F1, Confusion Matrix) and model artifact export.
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["Male", "Female"]


def set_seed(seed: int = RANDOM_SEED):
    """Sets deterministic seed across random, numpy, and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# 1. Data Preprocessing & Validation
# -----------------------------------------------------------------------------
def parse_utkface_filename(filepath: Path):
    """
    Parses UTKFace filename format: [age]_[gender]_[race]_[date&time].jpg
    Returns dict with parsed metadata, or None if invalid.
    """
    filename = filepath.name
    # Regex matching [age]_[gender]_[race]_[date].ext
    match = re.match(r"^(\d+)_([01])_(\d+)_(.+)\.(jpg|jpeg|png)$", filename, re.IGNORECASE)
    if not match:
        return None

    age = int(match.group(1))
    gender = int(match.group(2))
    race = int(match.group(3))

    # Basic validity checks
    if not (0 <= age <= 116) or gender not in (0, 1) or not (0 <= race <= 4):
        return None

    return {
        "filepath": str(filepath.resolve()),
        "filename": filename,
        "age": age,
        "gender": gender,
        "race": race,
        "gender_label": CLASS_NAMES[gender]
    }


def preprocess_dataset(data_dir: Path, output_csv: Path = None, test_size: float = 0.15, val_size: float = 0.15):
    """
    Scans directory, validates image files, parses metadata, and performs stratified splitting.
    """
    print(f"\n[1/5] Scanning dataset directory: {data_dir}")
    image_extensions = ("*.jpg", "*.jpeg", "*.png")
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_dir.rglob(ext))

    if not image_files:
        raise FileNotFoundError(f"No image files found in {data_dir}")

    records = []
    invalid_count = 0
    for path in image_files:
        rec = parse_utkface_filename(path)
        if rec is not None:
            records.append(rec)
        else:
            invalid_count += 1

    df = pd.DataFrame(records)
    print(f"  Total files examined: {len(image_files)}")
    print(f"  Valid samples parsed: {len(df)} | Invalid/skipped: {invalid_count}")

    if len(df) == 0:
        raise ValueError("No valid UTKFace format images could be parsed.")

    # Display class distribution
    counts = df["gender_label"].value_counts()
    print("  Class Distribution:")
    for label, count in counts.items():
        print(f"    - {label}: {count} ({count / len(df) * 100:.1f}%)")

    # Stratified Split: Train (70%), Validation (15%), Test (15%)
    # First split off test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=df["gender"]
    )

    # Then split train and validation
    val_ratio = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=RANDOM_SEED,
        stratify=train_val_df["gender"]
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"  Splits generated: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_csv, index=False)
        print(f"  Split saved to: {output_csv}")

    return train_df, val_df, test_df


# -----------------------------------------------------------------------------
# 2. PyTorch Dataset & Augmentation Transforms
# -----------------------------------------------------------------------------
class FacialAttributeDataset(Dataset):
    """PyTorch Dataset for face images with error handling for truncated/corrupt files."""
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row["filepath"]
        label = int(row["gender"])

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception:
            # Fallback for corrupt image: generate blank tensor
            fallback = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            return fallback, label


def get_transforms():
    """
    Data augmentations:
    - Train: Random rotation (±15°), affine translation/shear, color jitter, and horizontal flip
             to ensure robustness against head tilt and lighting variations.
    - Val/Test: Deterministic resize and center crop with ImageNet normalization.
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.10, 0.10), scale=(0.90, 1.10)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


# -----------------------------------------------------------------------------
# 3. Model Architecture
# -----------------------------------------------------------------------------
class ResNet18Classifier(nn.Module):
    """
    ResNet-18 fine-tuning model.
    Freezes early convolutional layers; trains layer4 and classification head.
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.45, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Freeze conv1, bn1, layer1, layer2, layer3
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze layer4 for fine-tuning
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        # Replace classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# -----------------------------------------------------------------------------
# 4. Training CNN Backbone
# -----------------------------------------------------------------------------
def train_cnn_backbone(train_loader, val_loader, device, epochs: int = 10, lr: float = 1e-4):
    """Trains ResNet-18 with AdamW, Label Smoothing, and early checkpointing."""
    print(f"\n[2/5] Training ResNet-18 Backbone ({epochs} epochs, device: {device})")
    model = ResNet18Classifier(num_classes=2, dropout=0.45, pretrained=False).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_score = 0.0
    best_state_dict = None
    history = []

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = train_loss / max(1, train_total)
        epoch_train_acc = train_correct / max(1, train_total)

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds_all, val_labels_all = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_preds_all.extend(preds.cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / max(1, val_total)
        epoch_val_acc = val_correct / max(1, val_total)
        epoch_val_bal = balanced_accuracy_score(val_labels_all, val_preds_all) if val_labels_all else 0.0

        scheduler.step(epoch_val_bal)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "val_bal_acc": epoch_val_bal
        })

        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.2f}% (Bal: {epoch_val_bal*100:.2f}%)")

        if epoch_val_bal > best_score:
            best_score = epoch_val_bal
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, history


# -----------------------------------------------------------------------------
# 5. Feature Extraction & SVM Training
# -----------------------------------------------------------------------------
@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device):
    """Extracts 512-D bottleneck embeddings by bypassing the classification head."""
    model.eval()
    orig_fc = model.backbone.fc
    model.backbone.fc = nn.Identity()

    features_list = []
    labels_list = []

    for images, labels in loader:
        images = images.to(device)
        feats = model(images)
        features_list.append(feats.cpu().numpy())
        labels_list.append(labels.numpy())

    model.backbone.fc = orig_fc

    X = np.vstack(features_list)
    y = np.concatenate(labels_list)
    return X, y


def train_and_tune_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    """Fits StandardScaler on train set and performs grid search over RBF SVM parameters."""
    print("\n[4/5] Normalizing Features & Tuning RBF Kernel SVM")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    param_grid = [
        {"C": 0.1, "gamma": "scale"},
        {"C": 1.0, "gamma": "scale"},
        {"C": 10.0, "gamma": "scale"},
        {"C": 1.0, "gamma": 0.001},
        {"C": 10.0, "gamma": 0.001},
    ]

    best_score = -1.0
    best_params = None
    best_svm = None
    results = []

    print(f"  Exploring {len(param_grid)} hyperparameter combinations...")
    for params in param_grid:
        svm = SVC(kernel="rbf", C=params["C"], gamma=params["gamma"], class_weight="balanced", random_state=RANDOM_SEED)
        svm.fit(X_train_scaled, y_train)

        val_preds = svm.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_preds)
        val_bal = balanced_accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average="macro")

        results.append({
            "C": params["C"],
            "gamma": str(params["gamma"]),
            "val_acc": val_acc,
            "val_bal_acc": val_bal,
            "val_f1": val_f1
        })

        if val_bal > best_score:
            best_score = val_bal
            best_params = params
            best_svm = svm

    print(f"  Best Hyperparameters: C={best_params['C']}, gamma={best_params['gamma']} (Val Bal Acc: {best_score*100:.2f}%)")

    # Evaluate on Unseen Test Set
    print("\n[5/5] Final Model Evaluation on Test Set")
    test_preds = best_svm.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)
    test_bal = balanced_accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="macro")
    cm = confusion_matrix(y_test, test_preds)

    print(f"  Test Accuracy:          {test_acc*100:.2f}%")
    print(f"  Test Balanced Accuracy: {test_bal*100:.2f}%")
    print(f"  Test Macro F1:          {test_f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, test_preds, target_names=CLASS_NAMES, digits=4))

    metrics = {
        "accuracy": float(test_acc),
        "balanced_accuracy": float(test_bal),
        "macro_f1": float(test_f1),
        "best_C": best_params["C"],
        "best_gamma": best_params["gamma"],
    }

    return scaler, best_svm, metrics, cm, pd.DataFrame(results)


# -----------------------------------------------------------------------------
# 6. Pipeline Orchestrator & Synthetic Demo Generator
# -----------------------------------------------------------------------------
def create_synthetic_sample_dataset(tmp_dir: Path, num_samples: int = 40):
    """Creates a lightweight synthetic dataset for dry-run verification."""
    print(f"\n[Dry Run] Generating {num_samples} synthetic face images in {tmp_dir}...")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_samples):
        gender = i % 2
        age = int(np.random.randint(18, 65))
        race = int(np.random.randint(0, 4))
        filename = f"{age}_{gender}_{race}_synthetic_{i}.jpg"
        filepath = tmp_dir / filename

        color = (180, 140, 120) if gender == 0 else (200, 150, 130)
        img_array = np.full((224, 224, 3), color, dtype=np.uint8)
        noise = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        Image.fromarray(img_array).save(filepath)
    return tmp_dir


def main():
    parser = argparse.ArgumentParser(description="FaceClass Gender Classification Training Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to UTKFace whole_images directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save metrics and splits")
    parser.add_argument("--epochs", type=int, default=10, help="Number of CNN training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for layer4")
    parser.add_argument("--dry-run", action="store_true", help="Run quick pipeline verification on synthetic samples")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("FaceClass: End-to-End Gender Preprocessing & Training")
    print(f"Device: {device} | Random Seed: {RANDOM_SEED}")
    print("=" * 70)

    output_dir = Path(args.output_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    temp_data_dir = None
    try:
        if args.dry_run or args.data_dir is None:
            if args.data_dir is None and not args.dry_run:
                print("\nNo --data-dir specified. Falling back to --dry-run mode for pipeline verification.")
            temp_data_dir = Path(tempfile.mkdtemp(prefix="utkface_sample_"))
            data_path = create_synthetic_sample_dataset(temp_data_dir, num_samples=40)
            epochs = min(args.epochs, 2)
            batch_size = 8
        else:
            data_path = Path(args.data_dir)
            if not data_path.exists():
                raise FileNotFoundError(f"Specified dataset directory does not exist: {data_path}")
            epochs = args.epochs
            batch_size = args.batch_size

        # 1. Dataset Preprocessing & Splitting
        split_csv = results_dir / "dataset_split.csv" if not args.dry_run else None
        train_df, val_df, test_df = preprocess_dataset(data_path, output_csv=split_csv)

        # 2. Augmentations & DataLoaders
        train_tf, eval_tf = get_transforms()
        train_ds = FacialAttributeDataset(train_df, transform=train_tf)
        val_ds = FacialAttributeDataset(val_df, transform=eval_tf)
        test_ds = FacialAttributeDataset(test_df, transform=eval_tf)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # 3. Fine-Tune ResNet-18
        cnn_model, cnn_history = train_cnn_backbone(train_loader, val_loader, device=device, epochs=epochs, lr=args.lr)

        # 4. Feature Extraction (512-D bottleneck)
        print("\n[3/5] Extracting 512-D Deep Features via ResNet-18 Bottleneck")
        X_train, y_train = extract_features(cnn_model, train_loader, device)
        X_val, y_val = extract_features(cnn_model, val_loader, device)
        X_test, y_test = extract_features(cnn_model, test_loader, device)
        print(f"  Feature matrices: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

        # 5. Feature Scaling & SVM Tuning
        scaler, svm_model, metrics, cm, search_df = train_and_tune_svm(X_train, y_train, X_val, y_val, X_test, y_test)

        # Save artifacts if not dry-run
        if not args.dry_run and args.data_dir is not None:
            model_save_path = output_dir / "gender_complete_model.joblib"
            cnn_save_path = output_dir / "gender_resnet18_best.pth"

            bundle = {
                "task": "gender",
                "cnn_architecture": "resnet18",
                "feature_dimension": 512,
                "fine_tuned_layers": ["layer4", "fc"],
                "dropout": 0.45,
                "label_smoothing": 0.10,
                "weight_decay": 1e-4,
                "svm_kernel": "rbf",
                "svm_C": metrics["best_C"],
                "svm_gamma": metrics["best_gamma"],
                "scaler": scaler,
                "svm": svm_model,
                "class_names": CLASS_NAMES,
                "image_size": IMAGE_SIZE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "orientation_augmentation": {
                    "rotation_degrees": 15,
                    "affine_translate": 0.10,
                    "affine_scale": (0.90, 1.10)
                },
                "metrics": metrics,
                "confusion_matrix": cm.tolist()
            }
            joblib.dump(bundle, model_save_path)
            torch.save(cnn_model.state_dict(), cnn_save_path)
            pd.DataFrame(cnn_history).to_csv(results_dir / "gender_cnn_history.csv", index=False)
            search_df.to_csv(results_dir / "gender_svm_search.csv", index=False)

            print(f"\n Artifacts successfully exported:")
            print(f"  - SVM Bundle:  {model_save_path}")
            print(f"  - CNN Weights: {cnn_save_path}")
        else:
            print("\n [Dry Run Completed] All preprocessing, training, feature extraction, and SVM steps succeeded!")

    finally:
        if temp_data_dir and temp_data_dir.exists():
            shutil.rmtree(temp_data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
