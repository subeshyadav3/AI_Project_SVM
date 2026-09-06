"""
Face Detection Module using OpenCV YuNet
YuNet is a lightweight, high-performance deep face detector (ONNX format)
capable of real-time detection on CPU.
"""

from pathlib import Path
import cv2
import numpy as np

DEFAULT_CONF_THRESHOLD = 0.60
DEFAULT_NMS_THRESHOLD = 0.30
DEFAULT_MIN_FACE_SIZE = 40  # Minimum face width/height in pixels

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
HAAR_MODEL_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"


class FaceDetector:
    """
    Detects human faces in an image using OpenCV YuNet.
    Falls back to Haar Cascade if the YuNet ONNX model is unavailable.
    """
    def __init__(self, conf_threshold: float = DEFAULT_CONF_THRESHOLD, min_face_size: int = DEFAULT_MIN_FACE_SIZE):
        self.conf_threshold = conf_threshold
        self.min_face_size = min_face_size
        self.yunet_detector = None
        self.haar_detector = None
        self.backend = "haar"

        # Initialize YuNet if ONNX model exists
        if YUNET_MODEL_PATH.exists():
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    model=str(YUNET_MODEL_PATH),
                    config="",
                    input_size=(320, 320),
                    score_threshold=self.conf_threshold,
                    nms_threshold=DEFAULT_NMS_THRESHOLD,
                    top_k=5000
                )
                self.backend = "yunet"
                print(f"[FaceDetector] Loaded YuNet face detector from {YUNET_MODEL_PATH.name}")
            except Exception as e:
                print(f"[FaceDetector] Failed to initialize YuNet: {e}")
                self.yunet_detector = None

        # Fallback to Haar cascade
        if self.yunet_detector is None and HAAR_MODEL_PATH.exists():
            try:
                self.haar_detector = cv2.CascadeClassifier(str(HAAR_MODEL_PATH))
                if not self.haar_detector.empty():
                    self.backend = "haar"
                    print(f"[FaceDetector] Loaded Haar cascade fallback from {HAAR_MODEL_PATH.name}")
            except Exception as e:
                print(f"[FaceDetector] Failed to load Haar cascade: {e}")

    def detect(self, image_bgr: np.ndarray, conf_threshold: float = None, min_face_size: int = None) -> list:
        """
        Detects faces in a BGR image.
        
        Returns:
            list of dicts: [{"bbox": [x, y, w, h], "confidence": float}, ...]
        """
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        min_size = min_face_size if min_face_size is not None else self.min_face_size

        img_h, img_w = image_bgr.shape[:2]
        faces = []

        if self.backend == "yunet" and self.yunet_detector is not None:
            # YuNet requires dynamic input dimensions per image
            self.yunet_detector.setInputSize((img_w, img_h))
            try:
                self.yunet_detector.setScoreThreshold(threshold)
            except Exception:
                pass

            _, detections = self.yunet_detector.detect(image_bgr)
            if detections is not None:
                for det in detections:
                    x, y, w, h = det[0:4].astype(int)
                    score = float(det[14])

                    if score < threshold:
                        continue
                    if w < min_size or h < min_size:
                        continue

                    # Clamp coordinates to image boundaries
                    x = max(0, min(x, img_w - 1))
                    y = max(0, min(y, img_h - 1))
                    w = min(w, img_w - x)
                    h = min(h, img_h - y)

                    if w > 0 and h > 0:
                        faces.append({
                            "bbox": [int(x), int(y), int(w), int(h)],
                            "confidence": score
                        })

        elif self.haar_detector is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            detected = self.haar_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_size, min_size)
            )
            for (x, y, w, h) in detected:
                faces.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "confidence": 0.85
                })

        return faces
