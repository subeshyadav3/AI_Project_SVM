"""
Gender Classification Model Pipeline
Architecture: ResNet-18 (512-D Feature Extractor) + RBF Kernel SVM Classifier
"""

from pathlib import Path
import cv2
import joblib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Resolve project paths
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
MODEL_DIR = REPO_ROOT / "models"

COMPLETE_MODEL_PATH = MODEL_DIR / "gender_complete_model.joblib"
RESNET_WEIGHTS_PATH = MODEL_DIR / "gender_resnet18_best.pth"

# Standard ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


class ResNet18FeatureExtractor(nn.Module):
    """
    ResNet-18 backbone fine-tuned for facial attribute extraction.
    We extract the 512-dimensional embedding from the penultimate average pooling layer.
    """
    def __init__(self, dropout: float = 0.45):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        # Classification head used during CNN fine-tuning
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to the global average pooling layer to extract 512-D features."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)


class GenderClassifier:
    """
    Two-stage inference pipeline:
    1. ResNet-18 extracts a 512-D deep feature vector from cropped face.
    2. StandardScaler normalizes features.
    3. RBF Kernel SVM classifies gender with a continuous decision margin.
    """
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GenderClassifier] Using device: {self.device}")

        # Check required model files
        if not COMPLETE_MODEL_PATH.exists():
            raise FileNotFoundError(f"SVM model bundle not found: {COMPLETE_MODEL_PATH}")
        if not RESNET_WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"ResNet weights not found: {RESNET_WEIGHTS_PATH}")

        # Load trained SVM, scaler, and training metadata
        print(f"[GenderClassifier] Loading SVM bundle from {COMPLETE_MODEL_PATH.name}...")
        bundle = joblib.load(COMPLETE_MODEL_PATH)
        self.scaler = bundle["scaler"]
        self.svm = bundle["svm"]
        self.class_names = bundle.get("class_names", ["Male", "Female"])
        self.image_size = bundle.get("image_size", IMAGE_SIZE)
        self.mean = bundle.get("mean", IMAGENET_MEAN)
        self.std = bundle.get("std", IMAGENET_STD)
        self.metrics = bundle.get("metrics", {})

        accuracy = self.metrics.get("accuracy", 0.0)
        print(f"[GenderClassifier] SVM loaded (C={bundle.get('svm_C')}, gamma={bundle.get('svm_gamma')}, Val Acc={accuracy*100:.2f}%)")

        # Load CNN feature extractor
        dropout = bundle.get("dropout", 0.45)
        self.cnn = ResNet18FeatureExtractor(dropout=dropout)
        state_dict = torch.load(RESNET_WEIGHTS_PATH, map_location=self.device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.cnn.load_state_dict(state_dict, strict=True)
        self.cnn.to(self.device)
        self.cnn.eval()
        print("[GenderClassifier] CNN backbone initialized in evaluation mode.")

        # Image preprocessing pipeline matching training
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    @torch.no_grad()
    def _extract_feature(self, pil_image: Image.Image) -> np.ndarray:
        """Extracts a (1, 512) normalized feature vector from a PIL face crop."""
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        features = self.cnn.extract_features(tensor)
        return features.cpu().numpy()

    def predict_pil(self, pil_image: Image.Image, gender_threshold: float = 0.0):
        """
        Predicts gender for a single cropped face.
        
        Parameters:
            pil_image: PIL Image of the detected face crop.
            gender_threshold: Decision boundary offset.
                              0.0 is the balanced mathematical optimum.
                              
        Returns:
            label: "Male" or "Female"
            confidence: Probability-like score between 0.50 and 0.99
            decision_value: Raw continuous SVM decision margin
        """
        # 1. Extract 512-D CNN features
        features = self._extract_feature(pil_image)

        # 2. Standard scale using training distribution
        features_scaled = self.scaler.transform(features)

        # 3. Compute continuous SVM margin distance
        try:
            decision = float(self.svm.decision_function(features_scaled)[0])
            pred_idx = 1 if decision > gender_threshold else 0

            # Convert margin to calibrated confidence using standard sigmoid mapping
            margin_diff = decision - gender_threshold
            if pred_idx == 1:
                confidence = 1.0 / (1.0 + np.exp(-margin_diff))
            else:
                confidence = 1.0 / (1.0 + np.exp(margin_diff))
            confidence = float(np.clip(confidence, 0.50, 0.99))
        except Exception:
            pred_idx = int(self.svm.predict(features_scaled)[0])
            decision = 0.0
            confidence = 0.92

        label = self.class_names[pred_idx]
        return label, confidence, decision

    def predict_faces(self, image_bgr: np.ndarray, faces: list, gender_threshold: float = 0.0) -> list:
        """
        Takes full BGR image and list of detected face boxes, crops each with context padding,
        and runs gender classification.
        """
        results = []
        img_h, img_w = image_bgr.shape[:2]

        for face in faces:
            x, y, w, h = face["bbox"]

            # Add 15% context padding around the face crop to capture hair and chin contours
            pad_w = int(w * 0.15)
            pad_h = int(h * 0.15)

            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(img_w, x + w + pad_w)
            y2 = min(img_h, y + h + pad_h)

            crop_bgr = image_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)

            label, confidence, decision = self.predict_pil(pil_crop, gender_threshold=gender_threshold)

            results.append({
                "bbox": face["bbox"],
                "padded_bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "detection_confidence": float(face.get("confidence", 0.95)),
                "gender": label,
                "confidence": float(confidence),
                "decision_value": float(decision),
            })

        return results
