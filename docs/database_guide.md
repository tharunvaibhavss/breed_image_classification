# PostgreSQL and SQLAlchemy Database Guide

This document specifies the relational database schema, ORM model entities, Alembic migration workflows, and database seeding procedures.

---

## 1. Relational Database Schema & Entities

The database schema is defined using SQLAlchemy 2.0 ORM in [`db/models.py`](file:///c:/Users/HP/Desktop/MCA%20Project/MCA%20Project%20AI%20Breed/db/models.py).

### Timezone-Aware UTC Datetime Handling
All SQLAlchemy model entities (`User`, `Breed`, `ImageMetadata`, `ModelVersion`, `Prediction`) utilize a centralized timezone-aware UTC datetime helper `utc_now()` (`datetime.datetime.now(datetime.timezone.utc)`). Deprecated `datetime.utcnow` calls have been fully removed to ensure complete forward-compatibility with Python 3.14+.

### Entity-Relationship Architecture

```
 ┌──────────────┐       1:N       ┌───────────────────┐
 │    users     │ ───────────────> │    predictions    │
 └──────────────┘                  └───────────────────┘
                                             │ N:1
 ┌──────────────┐                            │
 │    breeds    │ <──────────────────────────┤
 └──────────────┘                            │
                                             │ N:1
 ┌──────────────────────┐                    │
 │    image_metadata    │ <──────────────────┤
 └──────────────────────┘                    │
                                             │ N:1
 ┌──────────────────────┐                    │
 │    model_versions    │ <──────────────────┘
 └──────────────────────┘
```

---

## 2. Table Data Dictionary

### 1. `users` Table
- `id` (Integer, Primary Key)
- `email` (String, Unique, Indexed)
- `hashed_password` (String)
- `full_name` (String, Nullable)
- `is_active` (Boolean, Default: True)
- `is_superuser` (Boolean, Default: False)
- `created_at` (DateTime, UTC)

### 2. `breeds` Table
- `id` (Integer, Primary Key)
- `breed_name` (String, Unique, Indexed)
- `animal_type` (String, "cattle" or "buffalo")
- `origin` (String)
- `native_state` (String)
- `physical_characteristics` (JSON)
- `milk_production` (JSON)
- `climate_adaptability` (String)
- `uses` (String)
- `description` (Text)

### 3. `image_metadata` Table
- `id` (Integer, Primary Key)
- `filename` (String)
- `file_path` (String)
- `file_size_bytes` (Integer)
- `image_hash` (String, Indexed)
- `width` (Integer)
- `height` (Integer)

### 4. `model_versions` Table
- `id` (Integer, Primary Key)
- `model_name` (String)
- `yolo_version` (String)
- `efficientnet_version` (String)
- `gradcam_version` (String)
- `is_active` (Boolean, Default: True)

### 5. `predictions` Table
- `id` (Integer, Primary Key)
- `user_id` (Integer, Foreign Key $\rightarrow$ `users.id`)
- `image_id` (Integer, Foreign Key $\rightarrow$ `image_metadata.id`)
- `breed_id` (Integer, Foreign Key $\rightarrow$ `breeds.id`)
- `model_version_id` (Integer, Foreign Key $\rightarrow$ `model_versions.id`)
- `animal_type` (String)
- `animal_confidence` (Float)
- `bounding_box` (JSON)
- `predicted_breed_name` (String)
- `breed_confidence` (Float)
- `top_3_predictions` (JSON)
- `inference_time_ms` (JSON)
- `prediction_status` (String)

---

## 3. Alembic Migrations Workflow

```bash
# Create a new migration revision
alembic revision --autogenerate -m "Migration description"

# Upgrade database to latest revision
alembic upgrade head

# Rollback to previous migration
alembic downgrade -1
```

---

## 4. Seeding Initial Breed Records

Run the database seed script to populate the 6 default indigenous breed catalog entries:

```bash
python db/seed.py
```
