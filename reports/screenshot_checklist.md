# MCA Major Project — Screenshot & Visual Demonstration Checklist

**Project Title**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes Using Deep Learning  
**Application URL**: `http://localhost:3000` (Frontend) | `http://localhost:8000` (FastAPI Swagger Docs)  

---

## Visual Capture Checklist

| # | System Module | View / Route | Key Elements to Capture | Status |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Home / Landing Page** | `/` | Hero section, system architecture banner, CTA to image upload | Ready for capture |
| 2 | **User Registration** | `/register` | Registration form, full name, email, password strength indicator | Ready for capture |
| 3 | **User Login** | `/login` | JWT login form, secure credential input, validation error handling | Ready for capture |
| 4 | **Interactive Dashboard** | `/dashboard` | System statistics, recent scans, breed discovery shortcuts | Ready for capture |
| 5 | **Image Upload & Model Selection** | `/upload` | Drag-and-drop dropzone, model selector (82 vs 6 breeds), file size validation | Ready for capture |
| 6 | **Prediction Result (Cattle)** | `/upload` | Predicted breed (e.g. Gir Cattle), Top-1 confidence, YOLO detection box | Ready for capture |
| 7 | **Top-3 Candidates & Rankings** | `/upload` | Top-3 ranked breed candidate cards with dynamic progress bars | Ready for capture |
| 8 | **Grad-CAM Visual Heatmap** | `/upload` | 3-tab explainability viewer (Overlay, Heatmap, Original input) | Ready for capture |
| 9 | **Breed Catalog Explorer** | `/breeds` | Searchable grid of cattle and buffalo breeds with state of origin | Ready for capture |
| 10 | **Prediction History** | `/history` | Timestamped prediction log with thumbnail, predicted breed, confidence | Ready for capture |
| 11 | **User Profile View** | `/profile` | User identity details, authentication token state | Ready for capture |
| 12 | **Admin Dashboard** | `/admin` | System user management, database record counts | Ready for capture |
| 13 | **Model & System Info** | `/api/model-info` | Pipeline version, active PyTorch environment, checkpoints status | Ready for capture |
| 14 | **Swagger API Documentation** | `/docs` | Interactive OpenAPI documentation for all REST endpoints | Ready for capture |
| 15 | **Database Schema / Admin View** | SQLite DB / Alembic | Migrations table, predictions table, user accounts | Ready for capture |

---

## Pre-Generated High-Resolution Research Figures Available

The following publication-grade plots and demonstration figures have already been generated and saved in the project:
1. `plots/confusion_matrix.png` — Full 82x82 Breed Confusion Matrix
2. `plots/cattle_confusion_matrix.png` — 59-Cattle Breeds Confusion Matrix
3. `plots/buffalo_confusion_matrix.png` — 23-Buffalo Breeds Confusion Matrix
4. `plots/training_loss.png` — Training vs. Validation Loss Curves
5. `plots/training_accuracy.png` — Training vs. Validation Accuracy Curves
6. `plots/per_class_precision.png` — Per-Class Precision Distribution Chart
7. `results/gradcam/cattle_gir_gradcam.png` — Cattle Explainability Demonstration
8. `results/gradcam/buffalo_bhadawari_gradcam.png` — Buffalo Explainability Demonstration
