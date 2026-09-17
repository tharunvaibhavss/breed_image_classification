# REST API Documentation Specification

The FastAPI REST backend service exposes endpoints for health status, breed recognition prediction, breed catalogs, authentication, and administration.

---

## Base URL
`http://localhost:8000/api`

---

## Security & CORS Configuration
- **CORS Policy**: Configured via `CORS_ORIGINS` setting (`app/core/config.py`).
- **Development Mode**: Configured default origins: `["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"]`.
- **Production Mode**: Strictly enforces explicit domain origin arrays. Specifying wildcard `"*"` in production (`APP_ENV=production`) raises a startup `ValueError` to prevent insecure cross-origin requests.

---

## 1. System Health & Model Info Endpoints

### `GET /api/health`
- **Summary**: Health check endpoint.
- **Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "models": {
    "detection": "YOLO",
    "classification": "EfficientNet-B0",
    "explainability": "Grad-CAM"
  },
  "database": "connected"
}
```

### `GET /api/model-info`
- **Summary**: Hardware, PyTorch CUDA status, and model versions metadata.

---

## 2. Breed Recognition Prediction Endpoint

### `POST /api/predict`
- **Summary**: Upload livestock image and trigger full AI pipeline (YOLO $\rightarrow$ OpenCV $\rightarrow$ EfficientNet-B0 $\rightarrow$ Grad-CAM).
- **Request Format**: `multipart/form-data`
  - `file`: Image payload (max 10MB; JPEG, PNG, WebP, BMP, TIFF)
  - `generate_gradcam`: boolean (default `true`)
- **Response Schema**:
```json
{
  "animal_type": "cattle",
  "animal_confidence": 0.9452,
  "bounding_box": [25, 30, 350, 350],
  "predicted_breed": "Gir",
  "breed_confidence": 0.9481,
  "top_3_predictions": [
    {"class_id": 0, "breed_name": "Gir", "display_name": "Gir Cattle", "confidence": 0.9481},
    {"class_id": 2, "breed_name": "Sahiwal", "display_name": "Sahiwal Cattle", "confidence": 0.0321},
    {"class_id": 1, "breed_name": "Ongole", "display_name": "Ongole Cattle", "confidence": 0.0112}
  ],
  "prediction_status": "success",
  "inference_time": {
    "detection_ms": 12.5,
    "classification_ms": 14.8,
    "gradcam_ms": 15.2,
    "total_ms": 42.5
  },
  "model_versions": {
    "yolo_detector": "YOLOv8n-AnimalDetection-v1.0",
    "breed_classifier": "EfficientNet-B0-BreedRecognition-v1.0",
    "explainability": "Grad-CAM-v1.0"
  },
  "gradcam_output": {
    "display_name": "Gir Cattle",
    "target_class_id": 0,
    "overlay_base64": "data:image/jpeg;base64,...",
    "heatmap_base64": "data:image/jpeg;base64,..."
  }
}
```

---

## 3. Indigenous Breeds Catalog Endpoint

### `GET /api/breeds`
- **Summary**: Returns list of all 6 indigenous breed records with characteristics and milk yield metadata.

---

## 4. Authentication Endpoints (`/api/auth/*`)

- `POST /api/auth/register`: Register new user account.
- `POST /api/auth/login`: Authenticate and issue JWT Bearer access token.
- `POST /api/auth/logout`: Logout user session.
- `GET /api/auth/me`: Get current user profile details (Requires `Authorization: Bearer <token>`).

---

## 5. Administration Endpoints (`/api/admin/*`)
*Requires superuser administrative privileges (`is_superuser=True`).*

- `GET /api/admin/users`: List all registered user accounts.
- `PATCH /api/admin/users/{user_id}/status`: Enable/disable user accounts or toggle superuser privileges.
- `POST /api/admin/breeds`: Create new breed record.
- `GET /api/admin/analytics`: Aggregate prediction statistics and latency metrics.
- `GET /api/admin/model-versions`: Inspect active model versions.
- `GET /api/admin/dataset-metadata`: Inspect raw dataset version, split counts, and duplicate metrics.
