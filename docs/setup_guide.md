# System Setup and Installation Guide

This document provides complete instructions for setting up the development, testing, and production environments for the **AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes**.

---

## 1. System Requirements

- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Python**: Version 3.13.2 or 3.10+
- **Node.js**: Version 18+ (LTS)
- **Database**: PostgreSQL 15+ (or SQLite for local development)
- **RAM**: Minimum 8 GB (16 GB recommended)
- **Disk Space**: Minimum 5 GB free disk space

---

## 2. Environment Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/org/cattle-buffalo-breed-recognition.git
cd cattle-buffalo-breed-recognition
```

### Step 2: Python Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Configuration Management (`.env`)

Create a `.env` file in the root directory based on `.env.example`:

```ini
PROJECT_NAME="AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes"
APP_ENV=development
DEBUG=True

HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Security & Secrets
SECRET_KEY="your-secure-random-secret-key-string"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Database Configuration
DATABASE_URL="sqlite:///./breed_recognition_dev.db"
# For PostgreSQL: "postgresql://postgres:password@localhost:5432/breed_rec_db"

# Deep Learning Backend
INFERENCE_BACKEND=pytorch  # Options: 'pytorch' or 'onnx'
CUDA_VISIBLE_DEVICES=0
```

---

## 4. Database Setup & Migrations

### Run Database Migrations (Alembic)
```bash
# Apply migrations to database schema
alembic upgrade head

# Seed initial 6 indigenous breed records
python db/seed.py
```

---

## 5. Web Application Setup (`frontend/`)

```bash
cd frontend

# Install Node dependencies
npm install

# Run Next.js development server
npm run dev
```

The web interface will be accessible at `http://localhost:3000`.

---

## 6. Running the API Server

```bash
# Start FastAPI backend server with hot-reloading
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
