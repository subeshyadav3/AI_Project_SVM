import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
import cv2

# Paths - portable: works both locally and when repo is cloned
CURRENT_DIR = Path(__file__).parent.resolve()
# gender/ is sibling of Gender_classification_app/ in this repo branch
PROJECT_ROOT = CURRENT_DIR.parents[2] / "gender"
APP_MODEL_DIR = CURRENT_DIR / "models"

COMPLETE_MODEL_PATH = PROJECT_ROOT / "models" / "gender_complete_model.joblib"
RESNET_PTH = PROJECT_ROOT / "models" / "gender_resnet18_best.pth"

# Fallback if models copied into backend/models/
ALT_COMPLETE = APP_MODEL_DIR / "gender_complete_model.joblib"
ALT_PTH = APP_MODEL_DIR / "gender_resnet18_best.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, dropout=0.45):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        n_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(n_feat, 2))
    def forward(self, x):
        return self.backbone(x)
    def extract_features(self, x):
        x = self.backbone.conv1(x); x = self.backbone.bn1(x); x = self.backbone.relu(x)
        x = self.backbone.maxpool(x); x = self.backbone.layer1(x)
        x = self.backbone.layer2(x); x = self.backbone.layer3(x); x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x); x = torch.flatten(x, 1)
        return x

class GenderClassifier:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GenderClassifier] Device: {self.device}")

        # Resolve paths
        complete_path = COMPLETE_MODEL_PATH if COMPLETE_MODEL_PATH.exists() else ALT_COMPLETE
        pth_path = RESNET_PTH if RESNET_PTH.exists() else ALT_PTH

        if not complete_path.exists():
            raise FileNotFoundError(f"Complete model not found: {complete_path} / {ALT_COMPLETE}")
        if not pth_path.exists():
            raise FileNotFoundError(f"ResNet pth not found: {pth_path} / {ALT_PTH}")

        print(f"[GenderClassifier] Loading complete model: {complete_path}")
        bundle = joblib.load(complete_path)
        self.scaler = bundle["scaler"]
        self.svm = bundle["svm"]
        self.class_names = bundle.get("class_names", ["Male", "Female"])
        self.image_size = bundle.get("image_size", 224)
        self.mean = bundle.get("mean", IMAGENET_MEAN)
        self.std = bundle.get("std", IMAGENET_STD)
        self.metrics = bundle.get("metrics", {})
        print(f"[GenderClassifier] SVM: C={bundle.get('svm_C')} gamma={bundle.get('svm_gamma')} acc={self.metrics.get('accuracy', 'N/A')}")
        print(f"[GenderClassifier] Classes: {self.class_names}")

        # Load CNN feature extractor
        self.cnn = ResNet18FeatureExtractor(dropout=bundle.get("dropout", 0.45))
        state = torch.load(pth_path, map_location=self.device)
        # Handle both pure state_dict and checkpoint dict
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Remove possible 'backbone.' prefix mismatches already correct
        self.cnn.load_state_dict(state, strict=True)
        self.cnn.to(self.device)
        self.cnn.eval()
        print("[GenderClassifier] CNN loaded and in eval mode")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    @torch.no_grad()
    def _extract_feature(self, pil_image):
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        feat = self.cnn.extract_features(tensor)
        return feat.cpu().numpy()  # (1,512)

    def predict_pil(self, pil_image, gender_threshold=0.0):
        """
        Predict gender from PIL face crop.
        gender_threshold: decision value threshold for Female. dec > threshold => Female, else Male.
                          0.0 = default (prob 0.5). Raise to ~0.35-0.8 to reduce Male->Female false positives.
        Returns: (label_str, confidence_float, raw_decision)
        """
        feat = self._extract_feature(pil_image)  # (1,512)
        feat_scaled = self.scaler.transform(feat)

        # SVM decision function for confidence
        try:
            dec = self.svm.decision_function(feat_scaled)  # shape (n_samples,) for binary
            dec_val = float(dec[0])
            # thresholded prediction: dec > gender_threshold => Female else Male
            pred_idx = 1 if dec_val > gender_threshold else 0
            # confidence = sigmoid(|dec - threshold|) — distance from boundary
            if pred_idx == 1:
                confidence = float(1 / (1 + np.exp(-(dec_val - gender_threshold))))  # sigmoid(dec - thresh)
            else:
                confidence = float(1 / (1 + np.exp(dec_val - gender_threshold)))  # sigmoid(thresh - dec)
            confidence = max(0.5, min(0.99, confidence))
        except Exception as e:
            # Fallback to predict
            pred_idx = int(self.svm.predict(feat_scaled)[0])
            dec_val = 0.0
            confidence = 0.92 if pred_idx == 0 else 0.92

        label = self.class_names[pred_idx]
        return label, confidence, float(dec_val)

    def predict_faces(self, image_bgr, faces, gender_threshold=0.0):
        """
        image_bgr: full image (numpy BGR)
        faces: list from FaceDetector [{"bbox": [x,y,w,h], "confidence": score}]
        gender_threshold: see predict_pil
        Returns list with gender added
        """
        results = []
        for f in faces:
            x, y, w, h = f["bbox"]
            # Add slight padding (10%) for better context, clamp
            pad_w = int(w * 0.15)
            pad_h = int(h * 0.15)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(image_bgr.shape[1], x + w + pad_w)
            y2 = min(image_bgr.shape[0], y + h + pad_h)
            crop_bgr = image_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)

            label, conf, dec = self.predict_pil(pil_crop, gender_threshold=gender_threshold)
            results.append({
                "bbox": f["bbox"],
                "padded_bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "detection_confidence": float(f.get("confidence", 0.95)),
                "gender": label,
                "confidence": float(conf),
                "decision_value": float(dec)
            })
        return results
