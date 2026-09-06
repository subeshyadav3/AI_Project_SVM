"""
FaceClass - Facial Gender Classification Web Application
Backend: FastAPI + OpenCV YuNet Face Detection + ResNet18-SVM Classification
"""

from pathlib import Path
import base64
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from face_detector import DEFAULT_CONF_THRESHOLD, DEFAULT_MIN_FACE_SIZE, FaceDetector
from gender_model import GenderClassifier

app = FastAPI(
    title="FaceClass - Gender Classification System",
    description="Real-time Face Detection and Gender Classification using CNN (ResNet-18) and Support Vector Machines (SVM).",
    version="1.0.0"
)

# Enable CORS for local development and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model singletons
detector: FaceDetector = None
classifier: GenderClassifier = None


def get_detector() -> FaceDetector:
    global detector
    if detector is None:
        detector = FaceDetector(
            conf_threshold=DEFAULT_CONF_THRESHOLD,
            min_face_size=DEFAULT_MIN_FACE_SIZE
        )
    return detector


def get_classifier() -> GenderClassifier:
    global classifier
    if classifier is None:
        classifier = GenderClassifier()
    return classifier


@app.on_event("startup")
async def startup_event():
    """Pre-load models on server start so the first inference request is instant."""
    try:
        get_detector()
        get_classifier()
        print("[Server Startup] FaceDetector and GenderClassifier loaded successfully.")
    except Exception as e:
        print(f"[Server Startup Warning] Model loading error: {e}")


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "online",
        "detector": detector.backend if detector else "uninitialized",
        "classifier": "ready" if classifier else "uninitialized"
    }


@app.get("/config")
def get_configuration():
    """Returns current detector and classification thresholds."""
    det = get_detector()
    return {
        "detection_threshold": det.conf_threshold,
        "min_face_size": det.min_face_size,
        "gender_threshold": 0.0,
        "detector_backend": det.backend
    }


@app.post("/predict")
async def predict_gender(
    file: UploadFile = File(...),
    detection_threshold: float = Form(None),
    min_face_size: int = Form(None),
    gender_threshold: float = Form(None)
):
    """
    Main inference endpoint:
    1. Decodes uploaded image bytes.
    2. Detects human faces using YuNet.
    3. Extracts 512-D deep features via ResNet-18.
    4. Classifies gender using RBF SVM.
    5. Returns annotated bounding boxes, confidence, and base64 visualization.
    """
    try:
        det = get_detector()
        clf = get_classifier()

        # Parse threshold parameters with safe bounds
        conf_thresh = float(detection_threshold) if detection_threshold is not None else det.conf_threshold
        min_size = int(min_face_size) if min_face_size is not None else det.min_face_size
        g_thresh = float(gender_threshold) if gender_threshold is not None else 0.0

        conf_thresh = max(0.1, min(0.95, conf_thresh))
        min_size = max(20, min(300, min_size))
        g_thresh = max(-2.0, min(2.0, g_thresh))

        # Decode image from upload
        contents = await file.read()
        image_array = np.frombuffer(contents, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return JSONResponse(status_code=400, content={"error": "Could not decode uploaded image."})

        img_h, img_w = image_bgr.shape[:2]

        # 1. Detect faces
        faces = det.detect(image_bgr, conf_threshold=conf_thresh, min_face_size=min_size)

        if len(faces) == 0:
            _, buffer = cv2.imencode(".jpg", image_bgr)
            b64_image = base64.b64encode(buffer).decode("utf-8")
            return {
                "num_faces": 0,
                "faces": [],
                "image_width": img_w,
                "image_height": img_h,
                "annotated_image": f"data:image/jpeg;base64,{b64_image}",
                "message": "No faces detected. Try adjusting lighting or lowering detection threshold."
            }

        # 2. Classify detected faces
        results = clf.predict_faces(image_bgr, faces, gender_threshold=g_thresh)

        # 3. Draw clean visual annotations
        annotated = image_bgr.copy()
        for res in results:
            x, y, w, h = res["bbox"]
            label = res["gender"]
            conf = res["confidence"]

            # Visual color scheme: Male = Blue, Female = Pink
            box_color = (235, 99, 37) if label == "Male" else (180, 50, 220)  # BGR

            # Draw face bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)

            # Draw label tag
            tag_text = f"{label} {int(conf * 100)}%"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x, y - 26), (x + tw + 8, y), box_color, -1)
            cv2.putText(annotated, tag_text, (x + 4, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        _, buffer = cv2.imencode(".jpg", annotated)
        b64_annotated = base64.b64encode(buffer).decode("utf-8")

        return {
            "num_faces": len(results),
            "faces": results,
            "image_width": img_w,
            "image_height": img_h,
            "annotated_image": f"data:image/jpeg;base64,{b64_annotated}",
            "config_used": {
                "detection_threshold": conf_thresh,
                "min_face_size": min_size,
                "gender_threshold": g_thresh,
                "detector_backend": det.backend
            }
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Mount static frontend files if directory exists
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
