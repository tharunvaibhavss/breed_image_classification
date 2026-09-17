"""
Duplicate & Near-Duplicate Detection Engine.
Calculates MD5, SHA-256, pHash, and dHash across all images.
Identifies exact duplicates, resized copies, and near duplicates.
Segregates duplicates to dataset/rejected/duplicates/ and generates reports/duplicate_report.csv.
"""

import os
import sys
import csv
import shutil
import hashlib
from pathlib import Path
from PIL import Image
import imagehash

# Paths
WORKSPACE = Path(__file__).resolve().parent.parent
DATASET_ROOT = WORKSPACE / "dataset"
CLEANED_DIR = DATASET_ROOT / "cleaned"
EXPANDED_DIR = DATASET_ROOT / "expanded_82_breeds"
REJECTED_DUP_DIR = DATASET_ROOT / "rejected" / "duplicates"
REPORTS_DIR = DATASET_ROOT / "reports"
METADATA_DIR = DATASET_ROOT / "metadata"

DUPLICATE_REPORT_PATH = REPORTS_DIR / "duplicate_report.csv"
IMAGE_METADATA_PATH = METADATA_DIR / "image_metadata.csv"

# Thresholds
PHASH_HAMMING_THRESHOLD = 4  # pHash distance <= 4 indicates near-duplicate/resized copy
DHASH_HAMMING_THRESHOLD = 4


def calculate_hashes(filepath):
    """Calculates MD5, SHA-256, pHash, and dHash for an image file."""
    md5_h = hashlib.md5()
    sha_h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5_h.update(chunk)
            sha_h.update(chunk)
    
    with Image.open(filepath) as img:
        phash_val = str(imagehash.phash(img))
        dhash_val = str(imagehash.dhash(img))

    return md5_h.hexdigest(), sha_h.hexdigest(), phash_val, dhash_val


def run_duplicate_detection():
    print("=" * 70)
    print("STARTING DUPLICATE & NEAR-DUPLICATE DETECTION ENGINE")
    print("=" * 70)

    REJECTED_DUP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect all images in dataset/cleaned/
    image_entries = []
    for root, _, files in os.walk(CLEANED_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full_p = Path(root) / f
                try:
                    md5_val, sha_val, phash_val, dhash_val = calculate_hashes(full_p)
                    image_entries.append({
                        "path": full_p,
                        "filename": f,
                        "relative": str(full_p.relative_to(DATASET_ROOT)).replace("\\", "/"),
                        "species": full_p.parent.parent.name,
                        "breed_folder": full_p.parent.name,
                        "size": full_p.stat().st_size,
                        "md5": md5_val,
                        "sha256": sha_val,
                        "phash": phash_val,
                        "dhash": dhash_val
                    })
                except Exception as e:
                    print(f"Error reading {full_p}: {e}")

    print(f"Indexed {len(image_entries)} clean images for hash comparison.")

    duplicate_pairs = []
    removed_paths = set()

    # Compare all pairs
    for i in range(len(image_entries)):
        item1 = image_entries[i]
        if item1["path"] in removed_paths:
            continue

        for j in range(i + 1, len(image_entries)):
            item2 = image_entries[j]
            if item2["path"] in removed_paths:
                continue

            # Check exact duplicate
            if item1["sha256"] == item2["sha256"] or item1["md5"] == item2["md5"]:
                dup_record = {
                    "image_id_1": item1["filename"],
                    "image_id_2": item2["filename"],
                    "hash_type": "SHA-256/MD5",
                    "similarity": "1.000",
                    "duplicate_type": "exact_duplicate",
                    "action": "moved_to_rejected"
                }
                duplicate_pairs.append(dup_record)
                # Keep item1, remove item2
                removed_paths.add(item2["path"])
                continue

            # Check near-duplicate via perceptual hashes
            phash_dist = imagehash.hex_to_hash(item1["phash"]) - imagehash.hex_to_hash(item2["phash"])
            dhash_dist = imagehash.hex_to_hash(item1["dhash"]) - imagehash.hex_to_hash(item2["dhash"])

            if phash_dist <= PHASH_HAMMING_THRESHOLD or dhash_dist <= DHASH_HAMMING_THRESHOLD:
                sim_score = 1.0 - (min(phash_dist, dhash_dist) / 64.0)
                dup_record = {
                    "image_id_1": item1["filename"],
                    "image_id_2": item2["filename"],
                    "hash_type": f"pHash({phash_dist})/dHash({dhash_dist})",
                    "similarity": f"{sim_score:.3f}",
                    "duplicate_type": "near_duplicate_or_resized",
                    "action": "moved_to_rejected"
                }
                duplicate_pairs.append(dup_record)
                # Keep larger file, remove smaller
                if item1["size"] >= item2["size"]:
                    removed_paths.add(item2["path"])
                else:
                    removed_paths.add(item1["path"])
                    break  # item1 removed, break inner loop

    # Execute moves to rejected/duplicates/
    for p in removed_paths:
        dest = REJECTED_DUP_DIR / p.name
        # If filename collision in rejected, append hash
        if dest.exists():
            dest = REJECTED_DUP_DIR / f"{p.stem}_{hashlib.md5(p.name.encode()).hexdigest()[:6]}{p.suffix}"
        shutil.move(str(p), str(dest))

        # Also remove corresponding file in expanded_82_breeds if present
        exp_path = EXPANDED_DIR / p.parent.parent.name / p.parent.name / p.name
        if exp_path.exists():
            exp_path.unlink()

    # Write duplicate report CSV
    fields = ["image_id_1", "image_id_2", "hash_type", "similarity", "duplicate_type", "action"]
    with open(DUPLICATE_REPORT_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(duplicate_pairs)

    ROOT_DUPLICATE_REPORT_PATH = WORKSPACE / "reports" / "duplicate_report.csv"
    ROOT_DUPLICATE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ROOT_DUPLICATE_REPORT_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(duplicate_pairs)

    # Clean metadata/image_metadata.csv if duplicates were removed
    if removed_paths and IMAGE_METADATA_PATH.exists():
        cleaned_meta = []
        with open(IMAGE_METADATA_PATH, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            meta_fields = reader.fieldnames
            for row in reader:
                # check if filename is in removed_paths
                if not any(row["filename"] == p.name for p in removed_paths):
                    cleaned_meta.append(row)
        with open(IMAGE_METADATA_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=meta_fields)
            writer.writeheader()
            writer.writerows(cleaned_meta)

    print(f"\nDuplicate Detection Complete.")
    print(f"Total Duplicates Detected: {len(duplicate_pairs)}")
    print(f"Images moved to {REJECTED_DUP_DIR}: {len(removed_paths)}")
    print(f"Duplicate Report written to: {DUPLICATE_REPORT_PATH}")


if __name__ == "__main__":
    run_duplicate_detection()
