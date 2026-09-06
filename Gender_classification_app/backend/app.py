from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import io
from pathlib import Path
from PIL import Image
import uvicorn

from face_detector import FaceDetector, DEFAULT_CONF_THRESHOLD, DEFAULT_MIN_FACE_SIZE
from gender_model import GenderClassifier

app = FastAPI(title="Gender Classification App", version="1.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (lazy load)
detector = None
classifier = None

def get_detector():
    global detector
    if detector is None:
        detector = FaceDetector(conf_threshold=DEFAULT_CONF_THRESHOLD, min_face_size=DEFAULT_MIN_FACE_SIZE)
    return detector

def get_classifier():
    global classifier
    if classifier is None:
        classifier = GenderClassifier()
    return classifier

@app.on_event("startup")
async def startup():
    # Warm up
    try:
        get_detector()
        get_classifier()
        print("[Startup] Models loaded")
    except Exception as e:
        print(f"[Startup] Error loading models: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "detector": detector.backend if detector else "not_loaded", "classifier": "loaded" if classifier else "not_loaded"}

DEFAULT_GENDER_THRESHOLD = 0.0  # balanced optimum from val sweep: bal_acc 0.9237 at 0.0 (male_rec 0.9226 female_rec 0.9248 gap 0.002) — best balanced; 0.35 would drop test bal from 0.9195 to 0.9126
# global mutable gender threshold (shared)
GENDER_THRESHOLD = DEFAULT_GENDER_THRESHOLD

@app.get("/config")
def get_config():
    d = get_detector()
    return {
        "detection_threshold": d.conf_threshold,
        "min_face_size": d.min_face_size,
        "gender_threshold": GENDER_THRESHOLD,
        "defaults": {
            "detection_threshold": DEFAULT_CONF_THRESHOLD,
            "min_face_size": DEFAULT_MIN_FACE_SIZE,
            "gender_threshold": DEFAULT_GENDER_THRESHOLD,
            "note": "Best balanced: det 0.6, min 40px, gender 0.0 (val 92.37% bal, gap 0.002) — presentation-ready, no change needed"
        },
        "detector_backend": d.backend
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    detection_threshold: float = Form(None),
    min_face_size: int = Form(None),
    gender_threshold: float = Form(None)
):
    """
    Upload image, detect faces, classify gender.
    Accepts multipart/form-data with file + optional detection_threshold and min_face_size
    """
    try:
        # Use defaults if not provided
        det = get_detector()
        clf = get_classifier()

        thresh = float(detection_threshold) if detection_threshold is not None else det.conf_threshold
        min_size = int(min_face_size) if min_face_size is not None else det.min_face_size
        g_thresh = float(gender_threshold) if gender_threshold is not None else GENDER_THRESHOLD

        # Clamp
        thresh = max(0.0, min(0.99, thresh))
        min_size = max(10, min(300, min_size))
        g_thresh = max(-2.0, min(2.0, g_thresh))

        # Read image
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        h, w = img_bgr.shape[:2]

        # Detect faces
        faces = det.detect(img_bgr, conf_threshold=thresh, min_face_size=min_size)

        # Classify
        if len(faces) == 0:
            # Return with annotated image (no boxes)
            _, buffer = cv2.imencode('.jpg', img_bgr)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            return {
                "num_faces": 0,
                "faces": [],
                "image_width": w,
                "image_height": h,
                "annotated_image": f"data:image/jpeg;base64,{img_b64}",
                "config_used": {"detection_threshold": thresh, "min_face_size": min_size, "gender_threshold": g_thresh, "backend": det.backend},
                "message": "No faces detected. Try lowering detection threshold or min face size."
            }

        results = clf.predict_faces(img_bgr, faces, gender_threshold=g_thresh)

        # Draw annotated image
        annotated = img_bgr.copy()
        for r in results:
            x, y, ww, hh = r["bbox"]
            label = r["gender"]
            conf = r["confidence"]
            # Color: Male #2563eb, Female #db2777
            color = (235, 99, 37) if label == "Male" else (119, 39, 219)  # BGR
            cv2.rectangle(annotated, (x, y), (x+ww, y+hh), color, 2)
            text = f"{label} {conf:.2f}"
            # Background for text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x, y-28), (x+tw+8, y), color, -1)
            cv2.putText(annotated, text, (x+4, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', annotated)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        # Original also as base64 for frontend canvas option
        _, buf_orig = cv2.imencode('.jpg', img_bgr)
        orig_b64 = base64.b64encode(buf_orig).decode('utf-8')

        return {
            "num_faces": len(results),
            "faces": results,
            "image_width": w,
            "image_height": h,
            "annotated_image": f"data:image/jpeg;base64,{img_b64}",
            "original_image": f"data:image/jpeg;base64,{orig_b64}",
            "config_used": {"detection_threshold": thresh, "min_face_size": min_size, "gender_threshold": g_thresh, "backend": det.backend}
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# Serve frontend static if exists
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
