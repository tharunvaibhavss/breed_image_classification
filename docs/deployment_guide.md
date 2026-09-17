# Production Deployment Guide

This guide details production deployment procedures for the FastAPI ASGI application backend, Uvicorn/Gunicorn worker process management, PostgreSQL production configuration, Docker containerization, and Next.js frontend hosting.

---

## 1. Production Technology Stack Architecture

- **Web Server**: Nginx (Reverse Proxy & SSL Termination)
- **Application Server**: Gunicorn with Uvicorn Worker Processes (`uvicorn.workers.UvicornWorker`)
- **Database Server**: PostgreSQL 15+
- **Frontend App**: Next.js 14 Production Server (`npm run start` or Node.js Docker container)

---

## 2. Environment Configuration

Ensure production environment variables are configured in `/etc/environment` or `.env.production`:

```ini
APP_ENV=production
DEBUG=False
SECRET_KEY="production-high-entropy-crypto-secret-key"
DATABASE_URL="postgresql://breed_user:secure_prod_password@db.internal:5432/breed_rec_prod"
INFERENCE_BACKEND=pytorch  # or ONNX for optimized CPU hosting
ALLOWED_HOSTS=["breed-rec.example.com", "api.breed-rec.example.com"]
```

---

## 3. Production ASGI Server (Gunicorn + Uvicorn)

Run Gunicorn process manager with 4 worker processes:

```bash
gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile /var/log/gunicorn/access.log \
    --error-logfile /var/log/gunicorn/error.log
```

---

## 4. Docker Containerization Setup

### Dockerfile (FastAPI Backend)

```dockerfile
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
```

### Docker Compose Configuration (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  db:
    image: postgres:15-alpine
    restart: always
    environment:
      POSTGRES_USER: breed_user
      POSTGRES_PASSWORD: secure_prod_password
      POSTGRES_DB: breed_rec_prod
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  backend:
    build: .
    restart: always
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://breed_user:secure_prod_password@db:5432/breed_rec_prod
      - INFERENCE_BACKEND=pytorch
    depends_on:
      - db

  frontend:
    build: ./frontend
    restart: always
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000/api

volumes:
  postgres_data:
```

---

## 5. Next.js Web Frontend Build & Deployment

```bash
cd frontend
npm run build
npm run start
```
