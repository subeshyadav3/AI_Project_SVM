import cv2
import numpy as np
import os
import urllib.request
from pathlib import Path

# Default best configuration
DEFAULT_CONF_THRESHOLD = 0.6
DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_MIN_FACE_SIZE = 40  # minimum pixel size for face (width and height)
DEFAULT_TOP_K = 5000

HAARCASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

BACKEND_DIR = Path(__file__).parent
MODEL_DIR = BACKEND_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

HAAR_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"

def download_file(url, dest):
    try:
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def ensure_haar():
    if not HAAR_PATH.exists():
        download_file(HAARCASCADE_URL, HAAR_PATH)
    return HAAR_PATH.exists()

def ensure_yunet():
    if not YUNET_PATH.exists():
        download_file(YUNET_URL, YUNET_PATH)
    return YUNET_PATH.exists()

class FaceDetector:
    def __init__(self, conf_threshold=DEFAULT_CONF_THRESHOLD, min_face_size=DEFAULT_MIN_FACE_SIZE):
        self.conf_threshold = conf_threshold
        self.min_face_size = min_face_size
        self.yunet_detector = None
        self.haar_detector = None
        self.backend = "haar"

        # Try YuNet first (more accurate)
        if ensure_yunet():
            try:
                # YuNet needs input size, will be set per image
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    str(YUNET_PATH), "", (320, 320),
                    score_threshold=conf_threshold,
                    nms_threshold=DEFAULT_NMS_THRESHOLD,
                    top_k=DEFAULT_TOP_K
                )
                self.backend = "yunet"
                print(f"[FaceDetector] Using YuNet: {YUNET_PATH}")
            except Exception as e:
                print(f"[FaceDetector] YuNet init failed: {e}")
                self.yunet_detector = None

        if self.yunet_detector is None:
            if ensure_haar():
                try:
                    self.haar_detector = cv2.CascadeClassifier(str(HAAR_PATH))
                    if self.haar_detector.empty():
                        print("[FaceDetector] Haar cascade empty")
                        self.haar_detector = None
                    else:
                        self.backend = "haar"
                        print(f"[FaceDetector] Using Haar: {HAAR_PATH}")
                except Exception as e:
                    print(f"[FaceDetector] Haar init failed: {e}")
                    self.haar_detector = None

    def update_config(self, conf_threshold=None, min_face_size=None):
        if conf_threshold is not None:
            self.conf_threshold = float(conf_threshold)
            if self.yunet_detector is not None:
                # Recreate detector with new threshold (YuNet needs recreation or setScoreThreshold if available)
                try:
                    self.yunet_detector.setScoreThreshold(self.conf_threshold)
                except:
                    pass
        if min_face_size is not None:
            self.min_face_size = int(min_face_size)

    def detect(self, image_bgr, conf_threshold=None, min_face_size=None):
        """
        Detect faces in BGR image.
        Returns list of dicts: {bbox: [x,y,w,h], confidence: float}
        """
        if conf_threshold is not None:
            self.conf_threshold = float(conf_threshold)
        if min_face_size is not None:
            self.min_face_size = int(min_face_size)

        h, w = image_bgr.shape[:2]
        faces = []

        if self.backend == "yunet" and self.yunet_detector is not None:
            try:
                self.yunet_detector.setInputSize((w, h))
                # YuNet set threshold dynamically
                try:
                    self.yunet_detector.setScoreThreshold(self.conf_threshold)
                except:
                    pass
                _, results = self.yunet_detector.detect(image_bgr)
                if results is not None:
                    for det in results:
                        # det: [x, y, w, h, x_re, y_re, x_le, y_le, x_nose, y_nose, x_rmouth, y_rmouth, x_lmouth, y_lmouth, score]
                        x, y, ww, hh = det[0:4].astype(int)
                        score = float(det[14])
                        if score < self.conf_threshold:
                            continue
                        if ww < self.min_face_size or hh < self.min_face_size:
                            continue
                        # Clamp
                        x = max(0, x); y = max(0, y)
                        ww = min(ww, w - x); hh = min(hh, h - y)
                        if ww <= 0 or hh <= 0:
                            continue
                        faces.append({"bbox": [int(x), int(y), int(ww), int(hh)], "confidence": score})
                # If YuNet found faces, return
                if len(faces) > 0:
                    return faces
                # else fallback to Haar if YuNet found 0 but image may have faces YuNet missed at low res
            except Exception as e:
                print(f"[YuNet detect error] {e}")

        # Fallback / primary Haar
        if self.haar_detector is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            # Haar params tuned for best defaults
            detections = self.haar_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x, y, ww, hh) in detections:
                # Haar doesn't give confidence, assign 0.95 as pseudo
                # Filter by size already done
                faces.append({"bbox": [int(x), int(y), int(ww), int(hh)], "confidence": 0.95})

        # If still no faces and image is small, try with smaller minSize once
        if len(faces) == 0 and self.haar_detector is not None and self.min_face_size > 20:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            detections = self.haar_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
            for (x, y, ww, hh) in detections:
                if ww < self.min_face_size or hh < self.min_face_size:
                    continue
                faces.append({"bbox": [int(x), int(y), int(ww), int(hh)], "confidence": 0.85})

        return faces
