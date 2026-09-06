# Gender Classification — Branch `gender`

> **Standalone branch** for gender-only facial attribute classification.  
> This branch contains **only** `gender/` (training pipeline) + `Gender_classification_app/` (inference app). No code is shared with `main`.

Repo: `https://github.com/prabeshbchettri/FaceClass-CNN-SVM-Based-Facial-Attribute-Classification` — branch **`gender`**

---

## 1. What this branch is

| Folder | Purpose |
|---|---|
| `gender/` | Training notebook `SVM_GENDER_FIX.ipynb`, dataset split, saved models, evaluation CSVs |
| `Gender_classification_app/` | Production app: FastAPI backend + vanilla HTML/CSS/JS frontend. Runs completely offline after models download |

Both folders at repo root — no dependency on `main` branch code.

---

## 2. Architecture

### 2.1 High-level
```
Image (single or group photo)
   │
   ▼
[Face Detection]  YuNet (DNN) → fallback Haar Cascade
   │  conf_threshold=0.60, min_face_size=40px, NMS 0.3, topK 5000
   ▼
[Crop + Pad]  15% padding → Resize 256 → CenterCrop 224 → Normalize (ImageNet)
   │
   ▼
[ResNet18 Feature Extractor]  ImageNet mean/std, fine-tuned layer4+fc, dropout 0.45 → 512-D vector
   │
   ▼
[StandardScaler]  fitted on train features (joblib)
   │
   ▼
[RBF SVM]  C=0.01, gamma=scale → decision_value → threshold 0.0 → Male / Female + confidence (sigmoid)
   │
   ▼
 Annotated image + per-face JSON (bbox, gender, confidence, detection_confidence, decision_value)
```

### 2.2 Training pipeline (`gender/SVM_GENDER_FIX.ipynb` — 53 cells)

1. **Dataset** — UTKFace, filename encodes `age_gender_race_...jpg`. Parsed to `age, gender(0 Male/1 Female), race`. ~24k images.
2. **Split** — `dataset_split.csv` — stratified on **gender only** (fix vs earlier gender_race), train/val/test, deduplicated.
3. **Augmentation (orientation robustness)** — random ±15° rotation, affine, perspective to fix head-tilt failures.
4. **CNN training** — `ResNet18` pretrained → replace `fc → Dropout(0.45)+Linear(512→2)`, freeze early layers, train layer4+head with differential LR (`lr_layer4=1e-4, lr_head=5e-4`, decay on plateau). Best epoch tracked via `gender_cnn_history.csv` (val acc up to 92.36%).
5. **Feature extraction** — `extract_features()` (up to avgpool) on train/val/test → `features/gender_{train,val,test}.npz` (512-D), `gender_scaler.joblib`.
6. **SVM search** — grid `C ∈ {0.01,0.1,1,10} × gamma ∈ {scale,0.0005,0.001,0.005,0.01,0.1}` → `results/gender_svm_search.csv`. Best: **C=0.01 gamma=scale: val accuracy 92.33%, balanced 92.34%**.
7. **Export** — `models/gender_resnet18_best.pth` (CNN weights) + `gender_complete_model.joblib` (bundle: scaler+svm+class_names+mean/std+metrics). `checkpoints/gender_cnn_latest.pth` is training checkpoint (gitignored, >100MB).

### 2.3 Inference pipeline (`Gender_classification_app/`)

```
backend/
  app.py            FastAPI app, CORS *, lazy load models, /health, /config, /predict, serves frontend/
  face_detector.py  FaceDetector: tries YuNet ONNX (face_detection_yunet_2023mar.onnx), fallback Haar; auto-downloads if missing
  gender_model.py   GenderClassifier: loads bundle + pth, transform, predict_pil/predict_faces, portable paths (no hardcoded C:\...)
  requirements.txt  fastapi, uvicorn, torch, torchvision, scikit-learn, joblib, opencv-python, Pillow
  models/           face_detection_yunet_2023mar.onnx (auto-downloaded if absent)
frontend/
  index.html        upload, config modal (threshold sliders), results grid, info card
  script.js         drag&drop, FormData → /predict, render annotated image + per-face list, download
  style.css         minimal classic design, male #2563eb / female #db2777
run.bat             Windows one-click start
```

**Key parameters (tuned, presentation-ready):**

| Param | Default | Meaning |
|---|---|---|
| `detection_threshold` | `0.60` | YuNet score threshold, higher = stricter |
| `min_face_size` | `40` px | discard smaller boxes |
| `gender_threshold` | `0.0` | SVM decision boundary; `>0` biases to Male, `<0` to Female. 0.0 is best balanced (val 92.37% bal, gap 0.002) |

---

## 3. Models & Metrics

| Model file | Size | Role |
|---|---|---|
| `gender/models/gender_resnet18_best.pth` | 42.7 MB | CNN weights (ResNet18 fine-tuned) |
| `gender/models/gender_complete_model.joblib` | 21.9 MB | Bundle: scaler + SVM + metadata |
| `gender/models/gender_best_svm.joblib` | 18.8 MB | SVM only (redundant) |
| `gender/models/gender_final_svm.joblib` | 21.9 MB | SVM only (redundant) |
| `Gender_classification_app/backend/models/face_detection_yunet_2023mar.onnx` | 0.2 MB | YuNet face detector |

**Results:**

* CNN val best ~92.3% (`results/gender_cnn_history.csv` epoch 4: 92.36% val acc, 92.42% bal)
* SVM best C 0.01/scale: val accuracy 92.33%, macro F1 92.32% (`results/gender_svm_search.csv`)
* App threshold 0.0 → balanced optimum (male rec 92.26% female rec 92.48% gap 0.002, test bal 91.95%)

---

## 4. Quickstart

### 4.1 Clone this branch only
```bash
git clone --branch gender --single-branch https://github.com/prabeshbchettri/FaceClass-CNN-SVM-Based-Facial-Attribute-Classification.git
cd FaceClass-CNN-SVM-Based-Facial-Attribute-Classification
```

### 4.2 Run the app
```bat
REM Windows - double click:
Gender_classification_app\run.bat

REM or manually:
cd Gender_classification_app\backend
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000` — frontend is served from FastAPI at `/`.

### 4.3 API
```bash
GET  /health   # {status, detector backend, classifier}
GET  /config   # current thresholds + defaults

POST /predict  # multipart/form-data
# file: image (jpg/png)
# detection_threshold: float 0.0-0.99 (optional)
# min_face_size: int 10-300 (optional)
# gender_threshold: float -2..2 (optional)
```

### 4.4 Retrain
Open `gender/SVM_GENDER_FIX.ipynb` in Jupyter, set `DATASET_DIR`, run all cells. Outputs go to `gender/{models,features,results,checkpoints}`. Large artifacts `checkpoints/*.pth` and `features/*.npz` are gitignored — regenerate locally.

---

## 5. Repo structure (this branch)

```
.
├── README.md                          # this file (branch-specific)
├── .gitignore                         # ignores checkpoints/*.pth, features/*.npz, __pycache__, logs
├── gender/
│   ├── SVM_GENDER_FIX.ipynb
│   ├── dataset_split.csv              # 3.0 MB, stratified split
│   ├── checkpoints/.gitkeep           # (large .pth gitignored)
│   ├── features/
│   │   ├── gender_scaler.joblib
│   │   └── .gitkeep                   # (*.npz gitignored)
│   ├── models/
│   │   ├── gender_resnet18_best.pth
│   │   ├── gender_complete_model.joblib
│   │   ├── gender_best_svm.joblib
│   │   └── gender_final_svm.joblib
│   ├── results/
│   │   ├── gender_cnn_history.csv
│   │   └── gender_svm_search.csv
│   └── logs/
└── Gender_classification_app/
    ├── backend/
    │   ├── app.py
    │   ├── face_detector.py
    │   ├── gender_model.py            # portable paths (no hardcoded C:\)
    │   ├── requirements.txt
    │   └── models/
    │       └── face_detection_yunet_2023mar.onnx
    ├── frontend/
    │   ├── index.html
    │   ├── script.js
    │   └── style.css
    └── run.bat
```

---

## 6. Notes

* This branch is **orphan** — no history from `main`, so `git log` starts here.
* `gender_model.py` no longer contains hardcoded `C:\Users\LENOVO\...` — it resolves `PROJECT_ROOT = backend/../../gender`.
* YuNet ONNX auto-downloads from `opencv_zoo` if missing; Haar XML from `opencv` repo.
* Frontend tuning: detection_threshold 0.6 / min 40px / gender 0.0 are already optimal — no change needed for demo.

