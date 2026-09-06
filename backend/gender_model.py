"""
FaceClass - Gender Classification Model Pipeline
Architecture: ResNet-18 (512-D Feature Extractor) + StandardScaler + RBF Kernel SVM
"""

from pathlib import Path
import cv2
import joblib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Resolve model directory paths
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
COMPLETE_MODEL_PATH = MODEL_DIR / "gender_complete_model.joblib"
RESNET_WEIGHTS_PATH = MODEL_DIR / "gender_resnet18_best.pth"


class ResNet18FeatureExtractor(nn.Module):
    """
    ResNet-18 backbone with final classification layer replaced by Identity.
    Pass-through returns the 512-dimensional bottleneck embedding.
    """
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class GenderClassifier:
    """
    Two-stage inference pipeline:
    1. ResNet-18 extracts a 512-D deep feature vector from cropped face.
    2. StandardScaler normalizes features using training distribution statistics.
    3. RBF Kernel SVM classifies gender with continuous margin and calibrated confidence.
    """
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not COMPLETE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model bundle not found: {COMPLETE_MODEL_PATH}")
        if not RESNET_WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"ResNet weights not found: {RESNET_WEIGHTS_PATH}")

        # 1. Load trained SVM bundle (scaler, svm model, metadata)
        bundle = joblib.load(COMPLETE_MODEL_PATH)
        self.scaler = bundle["scaler"]
        self.svm = bundle["svm"]
        self.class_names = bundle.get("class_names", ["Male", "Female"])

        # 2. Load fine-tuned CNN backbone
        self.cnn = ResNet18FeatureExtractor()
        weights = torch.load(RESNET_WEIGHTS_PATH, map_location=self.device)
        self.cnn.load_state_dict(weights, strict=False)
        self.cnn.to(self.device).eval()

        # 3. Image preprocessing transform matching training
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=bundle.get("mean", [0.485, 0.456, 0.406]),
                std=bundle.get("std", [0.229, 0.224, 0.225])
            )
        ])

    @torch.no_grad()
    def _extract_feature(self, pil_image: Image.Image) -> np.ndarray:
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        return self.cnn(tensor).cpu().numpy()

    def predict_pil(self, pil_image: Image.Image, gender_threshold: float = 0.0):
        """
        Predicts gender for a single cropped face.
        Returns:
            label: 'Male' or 'Female'
            confidence: Float score in [0.50, 0.99]
            decision: Raw SVM margin distance
        """
        features = self._extract_feature(pil_image)
        features_scaled = self.scaler.transform(features)

        decision = float(self.svm.decision_function(features_scaled)[0])
        pred_idx = 1 if decision > gender_threshold else 0

        # Calibrate confidence from margin distance using sigmoid function
        margin_dist = abs(decision - gender_threshold)
        confidence = float(np.clip(1.0 / (1.0 + np.exp(-margin_dist)), 0.50, 0.99))

        return self.class_names[pred_idx], confidence, decision

    def predict_faces(self, image_bgr: np.ndarray, faces: list, gender_threshold: float = 0.0) -> list:
        """
        Processes full image and list of face bounding boxes with 15% contextual padding.
        """
        results = []
        img_h, img_w = image_bgr.shape[:2]

        for face in faces:
            x, y, w, h = face["bbox"]

            # Add 15% contextual margin to capture chin, hairstyle, and jawline cues
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
                "confidence": confidence,
                "decision_value": decision,
            })

        return results
