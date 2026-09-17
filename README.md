# AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14.2-black.svg)](https://nextjs.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.15+-blue.svg)](https://onnxruntime.ai/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791.svg)](https://www.postgresql.org/)
[![Build Status](https://img.shields.io/badge/Tests-82%20Passed-emerald.svg)]()

An end-to-end, production-grade Deep Learning system engineered for precise real-time detection, indigenous breed classification, and explainability heatmapping for Indian Cattle (*Bos indicus*) and Buffaloes (*Bubalus bubalis*).

---

## 🌟 Key System Capabilities

- **YOLOv8 Animal Detection**: Detects cattle (`class_id: 0`) and buffalo (`class_id: 1`) bounding box ROIs without breed classification.
- **EfficientNet-B0 Breed Classification**: Classifies crops into 6 indigenous Indian breeds (*Gir, Ongole, Sahiwal, Jaffarabadi, Murrah, Surti*) using transfer learning.
- **Grad-CAM Explainability**: Generates non-destructive visual attention heatmaps and JET color overlays highlighting facial and body features driving AI breed predictions.
- **ONNX Runtime Acceleration**: Provides a 3.04x speedup (14.59 ms ONNX CPU latency vs 44.40 ms PyTorch CPU latency) with 100% numerical prediction consistency.
- **FastAPI REST API Service**: Exposes validated endpoints with Pydantic DTO schemas, request tracking IDs, and error handling.
- **PostgreSQL Database Persistence**: Stores user accounts, breed catalogs, image metadata, and prediction history via SQLAlchemy 2.0 ORM & Alembic.
- **Modern Next.js 14 Web UI**: User-facing web application with glassmorphism UI design, drag-and-drop image analyzer, interactive Grad-CAM heatmap viewer, and history tracking.
- **Authentication & Security**: BCrypt password hashing, signed JWT Bearer access tokens, and Role-Based Access Control (`user` vs `administrator`).

---

## 🏛️ System Architecture Workflow

```
User Upload (Web App / API)
         │
         ▼
FastAPI Request ID Middleware
         │
         ▼
YOLOv8 Animal Detection (ROI Bounding Box)
         │
         ▼
OpenCV Preprocessing & Crop (224x224 RGB)
         │
         ▼
EfficientNet-B0 Breed Classification (PyTorch / ONNX Runtime)
         │
         ▼
Grad-CAM Visual Heatmap Hook (Explainability)
         │
         ▼
PostgreSQL Database Persistence & History Logging
         │
         ▼
JSON DTO Response + Next.js UI Display
```

---

## 🐂 Supported Indigenous Breeds Catalog (6 Classes)

| Class ID | Breed Name | Species | Native State | Primary Commercial Use |
| :--- | :--- | :--- | :--- | :--- |
| `0` | **Gir** | Cattle | Gujarat | Dairy (A2 Milk) |
| `1` | **Ongole** | Cattle | Andhra Pradesh | Dual-Purpose (Draft & Dairy) |
| `2` | **Sahiwal** | Cattle | Punjab | High-Yield Dairy |
| `3` | **Jaffarabadi** | Buffalo | Gujarat | High-Butterfat Dairy |
| `4` | **Murrah** | Buffalo | Haryana | Premier Dairy ("Black Gold") |
| `5` | **Surti** | Buffalo | Gujarat | Economical Dairy |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.13+
- Node.js 18+ and npm
- PostgreSQL database (optional for local SQLite mode)

### 2. Backend Setup
```bash
# Clone repository
git clone https://github.com/org/cattle-buffalo-breed-recognition.git
cd cattle-buffalo-breed-recognition

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run database migrations and seed default breeds
python db/seed.py

# Start FastAPI server
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Web Application Setup
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:3000` to open the web application.

---

## 🧪 Comprehensive Test Suite (82 Tests)

Run all unit, integration, API, security, and ONNX tests:
```bash
pytest
```
Run ONNX export and latency benchmark:
```bash
python -m scripts.export_and_benchmark_onnx
```

---

## 📚 Project Documentation Sitemap

Detailed documentation is available in the [`docs/`](file:///c:/Users/HP/Desktop/MCA%20Project/MCA%20Project%20AI%20Breed/docs/) directory:

- [Setup Guide](file:///c:/Users/HP/Desktop/MCA%20Project/MCA%20Project%20AI%20Breed/docs/setup_guide.md) — Installation, environment variables, database configuration.
- [Training Guide](file:///c:/Users/HP/Desktop/MCA%20Project/MCA%20Project%20AI%20Breed/docs/training_guide.md) — Training YOLO, Albumentations, and EfficientNet-B0 models.
- [Inference Guide](file:///c:/Users/HP/Desktop/MCA%20Project/MCA%20Project%20AI%20Breed/docs/inference_guide.md) — Executing Python, CLI, PyTorch, and ONNX Runtime inference.
- [API Documentation](file:///c:/Users/HP/Desktop/MCA%20Project/MCA%20Project%20AI%20Breed/docs/api_documentation.md) — OpenAPI endpoints specification & Pydantic schemas.
- [Database Guide](file:///c:/Users/HP/Desktop/MCA%20Project/MCA%20Project%20AI%20Breed/docs/database_guide.md) — SQLAlchemy ORM schema, ER diagram, Alembic migrations.
- [Deployment Guide](file:///c:/Users/HP/Desktop/MCA%20Project/MCA%20Project%20AI%20Breed/docs/deployment_guide.md) — Production ASGI/Uvicorn, Gunicorn, Docker, and Next.js setup.
- [Model Documentation](file:///c:/Users/HP/Desktop/MCA%20Project/MCA%20Project%20AI%20Breed/docs/model_documentation.md) — YOLOv8n, EfficientNet-B0, Grad-CAM, ONNX specs.
- [Dataset Documentation](file:///c:/Users/HP/Desktop/MCA%20Project/MCA%20Project%20AI%20Breed/docs/dataset_documentation.md) — 70/15/15 splitting, dHash deduplication, manifest format.
- [Final Testing Report](file:///c:/Users/HP/Desktop/MCA%20Project/MCA%20Project%20AI%20Breed/docs/final_testing_report.md) — Multi-dimensional test results & field condition matrix.
- [Known Limitations](file:///c:/Users/HP/Desktop/MCA%20Project/MCA%20Project%20AI%20Breed/docs/known_limitations.md) — Edge cases and operational boundaries.

---

## 📜 License & Acknowledgments

Engineered for ICAR and veterinary livestock researchers. Developed using PyTorch, OpenCV, FastAPI, Next.js, and ONNX Runtime.
