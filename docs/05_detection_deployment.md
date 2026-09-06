# 5. Face Detection & Real-Time Deployment Pipeline

## 5.1 Real-Time Face Detection via YuNet

In a production computer vision pipeline, raw input images contain full scenes, backgrounds, or multiple subjects. Accurate face localization is required before classification.

We deploy **OpenCV YuNet** (`models/face_detection_yunet_2023mar.onnx`), a lightweight, anchor-based convolutional face detector optimized for edge execution:

1. **Input Size Dynamism**: `cv2.FaceDetectorYN` dynamically adapts its receptive field to any input image dimension $(W, H)$.
2. **Confidence Thresholding ($\tau = 0.60$)**: Filters out weak background candidate detections.
3. **Non-Maximum Suppression (NMS, $\text{IoU} = 0.30$)**: Suppresses duplicate bounding boxes around the same facial region.
4. **Coordinate Clamping**: Clamps coordinates strictly within image bounds $[0, W-1]$ and $[0, H-1]$ to prevent index-out-of-bounds exceptions.

## 5.2 Contextual Face Cropping

Once bounding box $[x, y, w, h]$ is detected, simply cropping tightly around facial landmarks clips hair contours and jawlines, which are strong visual cues for gender determination.

We expand the bounding box with **15% contextual margin padding**:

$$x_1 = \max(0, x - 0.15w), \quad y_1 = \max(0, y - 0.15h)$$

$$x_2 = \min(W, x + 1.15w), \quad y_2 = \min(H, y + 1.15h)$$

The padded region is converted from OpenCV BGR to RGB and passed to the classification pipeline.

## 5.3 Complete Inference Pipeline

For every detected face:
1. **Crop**: Extract padded face ROI.
2. **Transform**: Deterministic resize $(256 \times 256) \to$ center crop $(224 \times 224) \to$ ImageNet channel normalization.
3. **Feature Extraction**: Feed through ResNet-18 up to the Global Average Pooling layer $\to \mathbf{z} \in \mathbb{R}^{512}$.
4. **Standardization**: Apply stored training scaler $\tilde{\mathbf{z}} = \frac{\mathbf{z} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$.
5. **SVM Decision**: Compute continuous decision margin $f(\tilde{\mathbf{z}}) = \sum_i \alpha_i y_i K(\mathbf{x}_i, \tilde{\mathbf{z}}) + b$.
6. **Confidence Calibration**: Map margin through sigmoid $\sigma(f(\tilde{\mathbf{z}}))$ to yield calibrated confidence.

### Latency Profile on Modern CPU
- Face Detection (YuNet): ~15–20 ms
- ResNet-18 Feature Extraction: ~10–12 ms
- SVM Margin Computation: ~1–2 ms
- **Total Pipeline Latency**: **< 35 ms per image** (enables real-time 30 FPS webcam processing).

## 5.4 Backend Architecture (FastAPI)

The backend (`backend/app.py`) provides high-performance asynchronous REST endpoints:
- `GET /health`: Model status and device verification.
- `GET /config`: Runtime detection and decision threshold settings.
- `POST /predict`: Multipart image upload returning JSON detections + annotated base64 visualization.
- Static mounting: Serves the web interface directly on the root path `/`.

## 5.5 Interactive Web Interface

The frontend (`frontend/index.html`, `script.js`, `style.css`) provides:
- Live webcam video capture with continuous frame inference.
- Drag-and-drop image file upload.
- Color-coded visual bounding boxes (Blue for Male, Pink for Female) with confidence badges.
- Real-time sliders for detection threshold and gender decision bias.
