"""
FaceClass - Face Detection Module
Uses OpenCV YuNet (fast ONNX detector) with automatic Haar Cascade fallback.
"""

from pathlib import Path
import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
HAAR_MODEL_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"


class FaceDetector:
    """
    Detects human faces in an image using OpenCV YuNet.
    Falls back to Haar Cascade if YuNet ONNX model is unavailable.
    """
    def __init__(self, conf_threshold: float = 0.60, min_face_size: int = 40):
        self.conf_threshold = conf_threshold
        self.min_face_size = min_face_size
        self.detector = None
        self.backend = "haar"

        if YUNET_MODEL_PATH.exists():
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=str(YUNET_MODEL_PATH),
                    config="",
                    input_size=(320, 320),
                    score_threshold=self.conf_threshold,
                    nms_threshold=0.30,
                    top_k=5000
                )
                self.backend = "yunet"
            except Exception:
                self.detector = None

        if self.detector is None and HAAR_MODEL_PATH.exists():
            cascade = cv2.CascadeClassifier(str(HAAR_MODEL_PATH))
            if not cascade.empty():
                self.detector = cascade
                self.backend = "haar"

    def detect(self, image_bgr: np.ndarray, conf_threshold: float = None, min_face_size: int = None) -> list:
        """
        Detects faces in BGR image.
        Returns:
            list of dicts: [{'bbox': [x, y, w, h], 'confidence': float}, ...]
        """
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        min_size = min_face_size if min_face_size is not None else self.min_face_size
        img_h, img_w = image_bgr.shape[:2]
        faces = []

        if self.backend == "yunet" and self.detector is not None:
            self.detector.setInputSize((img_w, img_h))
            self.detector.setScoreThreshold(threshold)
            _, detections = self.detector.detect(image_bgr)

            if detections is not None:
                for det in detections:
                    x, y, w, h = det[:4].astype(int)
                    score = float(det[14])
                    if score >= threshold and w >= min_size and h >= min_size:
                        # Clamp coordinates to image boundaries
                        x = max(0, min(x, img_w - 1))
                        y = max(0, min(y, img_h - 1))
                        w = min(w, img_w - x)
                        h = min(h, img_h - y)
                        if w > 0 and h > 0:
                            faces.append({"bbox": [int(x), int(y), int(w), int(h)], "confidence": score})

        elif self.detector is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            detections = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
            )
            for (x, y, w, h) in detections:
                faces.append({"bbox": [int(x), int(y), int(w), int(h)], "confidence": 0.85})

        return faces
