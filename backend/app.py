"""
FaceClass - Facial Gender Classification Web Application
Backend: FastAPI + OpenCV YuNet Detection + ResNet-18 & SVM Pipeline
"""

import sys
from pathlib import Path
import base64
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Support imports whether executed from root or backend directory
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from face_detector import FaceDetector
from gender_model import GenderClassifier

app = FastAPI(
    title="FaceClass - Gender Classification",
    description="Real-time Face Detection and Gender Classification using ResNet-18 and SVM.",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector and classifier singletons
detector = FaceDetector()
classifier = GenderClassifier()


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "online",
        "detector": detector.backend,
        "classifier": "ready"
    }


@app.get("/config")
def get_configuration():
    """Returns current detector and classification configuration."""
    return {
        "detection_threshold": detector.conf_threshold,
        "min_face_size": detector.min_face_size,
        "gender_threshold": 0.0,
        "detector_backend": detector.backend
    }


@app.post("/predict")
async def predict_gender(
    file: UploadFile = File(...),
    detection_threshold: float = Form(0.60),
    min_face_size: int = Form(40),
    gender_threshold: float = Form(0.0)
):
    """
    Main inference pipeline:
    1. Decodes uploaded image bytes.
    2. Detects human faces using YuNet.
    3. Extracts 512-D deep features via ResNet-18.
    4. Predicts gender using RBF SVM.
    5. Returns annotated bounding boxes, confidence scores, and visual base64 image.
    """
    try:
        contents = await file.read()
        image_array = np.frombuffer(contents, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file format."})

        img_h, img_w = image_bgr.shape[:2]

        # 1. Detect faces
        faces = detector.detect(image_bgr, conf_threshold=detection_threshold, min_face_size=min_face_size)
        if not faces:
            _, buffer = cv2.imencode(".jpg", image_bgr)
            b64_img = base64.b64encode(buffer).decode("utf-8")
            return {
                "num_faces": 0,
                "faces": [],
                "image_width": img_w,
                "image_height": img_h,
                "annotated_image": f"data:image/jpeg;base64,{b64_img}",
                "message": "No faces detected."
            }

        # 2. Classify gender for detected faces
        results = classifier.predict_faces(image_bgr, faces, gender_threshold=gender_threshold)

        # 3. Draw clean visual annotations
        annotated = image_bgr.copy()
        for res in results:
            x, y, w, h = res["bbox"]
            label = res["gender"]
            conf = res["confidence"]

            # Visual color scheme: Male = Blue/Cyan, Female = Magenta/Pink
            color = (235, 99, 37) if label == "Male" else (180, 50, 220)  # BGR

            # Bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Label tag
            tag = f"{label} {int(conf * 100)}%"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x, y - 26), (x + tw + 8, y), color, -1)
            cv2.putText(annotated, tag, (x + 4, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        _, buffer = cv2.imencode(".jpg", annotated)
        b64_annotated = base64.b64encode(buffer).decode("utf-8")

        return {
            "num_faces": len(results),
            "faces": results,
            "image_width": img_w,
            "image_height": img_h,
            "annotated_image": f"data:image/jpeg;base64,{b64_annotated}",
            "config_used": {
                "detection_threshold": detection_threshold,
                "min_face_size": min_face_size,
                "gender_threshold": gender_threshold,
                "detector_backend": detector.backend
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Mount static frontend directory
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, app_dir=str(CURRENT_DIR))
