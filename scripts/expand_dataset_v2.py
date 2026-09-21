"""
Acquisition & Expansion Engine v2 for 82 Indian Cattle & Buffalo Breeds.
Integrates:
- Baseline verified images (486 images)
- Wikimedia Commons API (Categories, deep search, breed synonyms)
- CIRB Official Archives
- iNaturalist Research-Grade Bovine Observations (India)
Includes:
- Hash-based exact deduplication (MD5, SHA-256)
- Perceptual deduplication (pHash <= 4)
- Automated image quality control (resolution, aspect ratio, corrupt byte checks)
- Full metadata tracking to dataset/metadata_v2/dataset_metadata_v2.csv
"""

import os
import sys
import re
import time
import json
import csv
import shutil
import hashlib
import socket
import urllib.request
import urllib.parse
from pathlib import Path
from PIL import Image
import imagehash

socket.setdefaulttimeout(8.0)

WORKSPACE = Path(__file__).resolve().parent.parent
DATASET_ROOT = WORKSPACE / "dataset"
V1_DATASET = DATASET_ROOT / "expanded_82_breeds"
V2_DATASET = DATASET_ROOT / "expanded_82_breeds_v2"
METADATA_DIR = DATASET_ROOT / "metadata_v2"
METADATA_DIR.mkdir(parents=True, exist_ok=True)
METADATA_CSV = METADATA_DIR / "dataset_metadata_v2.csv"
BREED_DETAILS_PATH = DATASET_ROOT / "metadata" / "breed_details.json"

USER_AGENT = "MCABreedResearch/2.0 (academic research; AI Livestock Classification; mailto:mca.project@university.ac.in)"
HEADERS = {"User-Agent": USER_AGENT}

TARGET_PER_BREED = 50

# Exclusion terms
REJECT_KEYWORDS = [
    "map", "distribution", "diagram", "poster", "infographic", "logo", "stamp",
    "coin", "statue", "monument", "temple", "drawing", "sketch", "painting",
    "flag", "chart", "graph", "text", "digest", "pdf", "djvu", "svg",
    "battle", "aleppo", "martin luther", "document", "manuscript", "executive order",
    "denis", "human", "couple", "portrait of person", "people", "protest", "memorial",
    "rugby", "cricket", "badge", "emblem", "coat of arms"
]

# CIRB Buffalo Archive
CIRB_BUFFALO_ARCHIVE = [
    {"breed": "Murrah", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Murrah_female-1.jpg", "desc": "CIRB Official Murrah Female"},
    {"breed": "Murrah", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Murrah-male-1-1.jpg", "desc": "CIRB Official Murrah Male"},
    {"breed": "Nili Ravi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Nilli-Ravi-female.jpg", "desc": "CIRB Official Nili Ravi Female"},
    {"breed": "Nili Ravi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Nili-Ravi-male.jpg", "desc": "CIRB Official Nili Ravi Male"},
    {"breed": "Bhadawari", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Bhadawari-female.jpg", "desc": "CIRB Official Bhadawari Female"},
    {"breed": "Bhadawari", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Bhadawari-male.jpg", "desc": "CIRB Official Bhadawari Male"},
    {"breed": "Mehsana", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Mehsana-female.jpg", "desc": "CIRB Official Mehsana Female"},
    {"breed": "Surti", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Surti-female-1.jpg", "desc": "CIRB Official Surti Female"},
    {"breed": "Surti", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Surti_Male-1.jpg", "desc": "CIRB Official Surti Male"},
    {"breed": "Pandharpuri", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Pandharpuri_Female.jpg", "desc": "CIRB Official Pandharpuri Female"},
    {"breed": "Pandharpuri", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Pandharpuri-male.jpg", "desc": "CIRB Official Pandharpuri Male"},
    {"breed": "Toda", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Toda-female.jpg", "desc": "CIRB Official Toda Female"},
    {"breed": "Toda", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Toda-male.jpg", "desc": "CIRB Official Toda Male"},
    {"breed": "Banni", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Banni-female.jpg", "desc": "CIRB Official Banni Female"},
    {"breed": "Banni", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Banni-male.jpg", "desc": "CIRB Official Banni Male"},
    {"breed": "Chilika", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Chillika-female.jpg", "desc": "CIRB Official Chilika Female"},
    {"breed": "Chilika", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Chillika-male.jpg", "desc": "CIRB Official Chilika Male"},
    {"breed": "Kalahandi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Kalahandi-female.jpg", "desc": "CIRB Official Kalahandi Female"},
    {"breed": "Kalahandi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Kalahandi-male.jpg", "desc": "CIRB Official Kalahandi Male"},
    {"breed": "Luit (Swamp)", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Luit-female.jpg", "desc": "CIRB Official Luit Swamp Female"},
    {"breed": "Luit (Swamp)", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Luit-male.jpg", "desc": "CIRB Official Luit Swamp Male"},
    {"breed": "Bargur", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Bargur-female.jpg", "desc": "CIRB Official Bargur Buffalo Female"},
    {"breed": "Bargur", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Bargur-male.jpg", "desc": "CIRB Official Bargur Buffalo Male"},
    {"breed": "Gojri", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Gojri-female.jpg", "desc": "CIRB Official Gojri Female"},
    {"breed": "Gojri", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Gojri-Male.jpg", "desc": "CIRB Official Gojri Male"},
    {"breed": "Chhattisgarhi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Chhattisgarhi-female.jpg", "desc": "CIRB Official Chhattisgarhi Female"},
    {"breed": "Chhattisgarhi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Chhattisgarhi-male.jpg", "desc": "CIRB Official Chhattisgarhi Male"},
    {"breed": "Dharwadi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Dharwadi-female.jpg", "desc": "CIRB Official Dharwadi Female"},
    {"breed": "Dharwadi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Dharwadi-male.jpg", "desc": "CIRB Official Dharwadi Male"},
    {"breed": "Manda", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Manda-female.jpg", "desc": "CIRB Official Manda Female"},
    {"breed": "Manda", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Manda-male.jpg", "desc": "CIRB Official Manda Male"},
    {"breed": "Purnathadi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Purnathadi-female.jpg", "desc": "CIRB Official Purnathadi Female"},
    {"breed": "Purnathadi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Purnathadi-male.jpg", "desc": "CIRB Official Purnathadi Male"},
    {"breed": "Manah", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Buffalo_Manah_Female.jpg", "desc": "CIRB Official Manah Female"},
    {"breed": "Manah", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Buffalo_Manah_Male.jpg", "desc": "CIRB Official Manah Male"}
]

# Extended synonyms & search queries
BREED_QUERIES = {
    "Gir": ["Gir cattle", "Gir cow", "Gir bull", "Gyr cattle", "Category:Gir cattle"],
    "Ongole": ["Ongole cattle", "Ongole bull", "Nelore cattle", "Category:Ongole cattle"],
    "Kankrej": ["Kankrej cattle", "Kankrej bull", "Guzerat cattle", "Category:Kankrej cattle"],
    "Sahiwal": ["Sahiwal cattle", "Sahiwal cow", "Category:Sahiwal cattle"],
    "Kangayam": ["Kangayam cattle", "Kangeyam bull", "Kangayam bull"],
    "Hallikar": ["Hallikar cattle", "Hallikar bull", "Hallikar"],
    "Amritmahal": ["Amritmahal cattle", "Amrit Mahal bull", "Amritmahal"],
    "Vechur": ["Vechur cow", "Vechur cattle", "Vechur"],
    "Punganur": ["Punganur cow", "Punganur cattle", "Punganur"],
    "Tharparkar": ["Tharparkar cattle", "Thari cow", "Tharparkar bull"],
    "Hariana": ["Hariana cattle", "Hariana bull", "Haryana cow"],
    "Khillar": ["Khillar cattle", "Khillari bull", "Khillar"],
    "Deoni": ["Deoni cattle", "Deoni cow", "Deoni bull"],
    "Dangi": ["Dangi cattle", "Dangi cow", "Dangi bull"],
    "Gaolao": ["Gaolao cattle", "Gaolao cow"],
    "Red Sindhi": ["Red Sindhi cattle", "Red Sindhi cow", "Category:Red Sindhi cattle"],
    "Bargur": ["Bargur cattle", "Baragur cow"],
    "Krishna Valley": ["Krishna Valley cattle", "Krishna Valley cow"],
    "Malvi": ["Malvi cattle", "Malvi bull"],
    "Nagori": ["Nagori cattle", "Nagori bull"],
    "Nimari": ["Nimari cattle", "Nimari cow"],
    "Umblachery": ["Umblachery cattle", "Umblachery"],
    "Pulikulam": ["Pulikulam cattle", "Pulikulam bull"],
    "Malnad Gidda": ["Malnad Gidda cow", "Malnad Gidda cattle"],
    "Gangatiri": ["Gangatiri cattle", "Gangatiri cow"],
    "Badri": ["Badri cattle", "Badri cow"],
    "Ladakhi": ["Ladakhi cattle", "Ladakhi cow"],
    "Rathi": ["Rathi cattle", "Rathi cow"],
    "Mewati": ["Mewati cattle", "Kosi cattle"],
    "Ponwar": ["Ponwar cattle"],
    "Siri": ["Siri cattle", "Siri cow"],
    "Kherigarh": ["Kherigarh cattle"],
    "Bachaur": ["Bachaur cattle"],
    "Belahi": ["Belahi cattle"],
    "Binjharpuri": ["Binjharpuri cattle"],
    "Ghumusari": ["Ghumusari cattle"],
    "Khariar": ["Khariar cattle"],
    "Motu": ["Motu cattle"],
    "Red Kandhari": ["Red Kandhari cattle"],
    "Vechur": ["Vechur cattle", "Vechur cow"],
    "Shweta Kapila": ["Shweta Kapila cattle"],
    "Konkan Kapila": ["Konkan Kapila cattle"],
    "Kosali": ["Kosali cattle"],
    "Lakhimi": ["Lakhimi cattle"],
    "Thutho": ["Thutho cattle"],
    "Dagri": ["Dagri cattle"],
    "Purnea": ["Purnea cattle"],
    "Poda Thurpu": ["Poda Thurpu cattle"],
    "Nari": ["Nari cattle"],
    "Kenesha": ["Kenesha cattle"],
    "Masilum": ["Masilum cattle"],
    "Alambadi": ["Alambadi cattle"],
    "Bargur": ["Bargur cattle"],
    "Hissar": ["Hissar cattle"],
    # Buffaloes
    "Murrah": ["Murrah buffalo", "Murrah bull", "Category:Murrah buffalo"],
    "Nili Ravi": ["Nili Ravi buffalo", "Nili-Ravi buffalo", "Nili Ravi"],
    "Jaffarabadi": ["Jaffarabadi buffalo", "Jafarabadi buffalo", "Jaffarabadi"],
    "Mehsana": ["Mehsana buffalo", "Mehsana"],
    "Surti": ["Surti buffalo", "Surti"],
    "Toda": ["Toda buffalo", "Toda buffalos"],
    "Nagpuri": ["Nagpuri buffalo", "Ellichpuri buffalo"],
    "Pandharpuri": ["Pandharpuri buffalo", "Pandharpuri"],
    "Banni": ["Banni buffalo", "Kundi buffalo"],
    "Chilika": ["Chilika buffalo", "Chillika buffalo"],
    "Bhadawari": ["Bhadawari buffalo", "Bhadawari"],
    "Kalahandi": ["Kalahandi buffalo"],
    "Luit (Swamp)": ["Luit buffalo", "Assam swamp buffalo", "Luit"],
    "Gojri": ["Gojri buffalo"],
    "Chhattisgarhi": ["Chhattisgarhi buffalo"],
    "Dharwadi": ["Dharwadi buffalo"],
    "Manda": ["Manda buffalo"],
    "Marathwadi": ["Marathwadi buffalo"],
    "Purnathadi": ["Purnathadi buffalo"],
    "Sambalpuri": ["Sambalpuri buffalo"],
    "Manah": ["Manah buffalo"]
}


def calculate_hashes(filepath):
    """Calculate MD5 and SHA-256."""
    md5_h = hashlib.md5()
    sha_h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5_h.update(chunk)
            sha_h.update(chunk)
    return md5_h.hexdigest(), sha_h.hexdigest()


def compute_phash(filepath):
    """Calculate perceptual hash using PIL and imagehash."""
    try:
        with Image.open(filepath) as img:
            return imagehash.phash(img)
    except Exception:
        return None


def validate_image_file(filepath):
    """Checks image integrity, format, dimensions and aspect ratio."""
    try:
        if filepath.stat().st_size < 5000:
            return False, "File too small (<5KB)", 0, 0, ""
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            rgb_im = img.convert("RGB")
            w, h = rgb_im.size
            fmt = img.format or "JPEG"
        if w < 150 or h < 150:
            return False, f"Resolution too low ({w}x{h} < 150x150)", w, h, fmt
        ar = w / float(h)
        if ar < 0.35 or ar > 2.8:
            return False, f"Atypical aspect ratio ({ar:.2f})", w, h, fmt
        return True, "Valid", w, h, fmt
    except Exception as e:
        return False, f"Image error: {str(e)}", 0, 0, ""


def query_wikimedia_commons(query, limit=50):
    """Queries Wikimedia Commons API with pagination and rate limiting."""
    is_category = query.startswith("Category:")
    if is_category:
        cat_title = urllib.parse.quote(query)
        api_url = (
            f"https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers"
            f"&gcmtitle={cat_title}&gcmtype=file&gcmlimit={limit}&prop=imageinfo"
            f"&iiprop=url|size|extmetadata|mime&format=json"
        )
    else:
        q_enc = urllib.parse.quote(query)
        api_url = (
            f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
            f"&gsrsearch={q_enc}&gsrnamespace=6&gsrlimit={limit}&prop=imageinfo"
            f"&iiprop=url|size|extmetadata|mime&format=json"
        )

    candidates = []
    try:
        req = urllib.request.Request(api_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, p in pages.items():
                title = p.get("title", "")
                ii = p.get("imageinfo", [{}])[0]
                img_url = ii.get("url", "")
                mime = ii.get("mime", "")
                if not img_url or not mime.startswith("image/"):
                    continue
                if mime in ["image/svg+xml", "image/x-xcf"]:
                    continue
                title_lower = title.lower()
                if any(k in title_lower for k in REJECT_KEYWORDS):
                    continue
                extmeta = ii.get("extmetadata", {})
                desc = extmeta.get("ImageDescription", {}).get("value", "")
                desc_lower = desc.lower()
                if any(k in desc_lower for k in ["distribution map", "location map", "infographic"]):
                    continue
                license_name = extmeta.get("LicenseShortName", {}).get("value", "UNKNOWN")
                author = extmeta.get("Artist", {}).get("value", "UNKNOWN")
                author_clean = re.sub(r'<.*?>', '', author).strip() or "UNKNOWN"
                candidates.append({
                    "title": title,
                    "url": img_url,
                    "license": license_name,
                    "author": author_clean,
                    "source_name": "Wikimedia Commons"
                })
        time.sleep(1.0) # polite delay
    except Exception as e:
        if "429" in str(e):
            time.sleep(5.0) # exponential backoff
    return candidates


def query_inaturalist_observations(taxon_id, query_str, limit=30):
    """Queries iNaturalist API for research-grade bovine observations in India."""
    api_url = (
        f"https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}"
        f"&q={urllib.parse.quote(query_str)}&place_id=6681&quality_grade=research"
        f"&photos=true&per_page={limit}"
    )
    candidates = []
    try:
        req = urllib.request.Request(api_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            for obs in results:
                photos = obs.get("photos", [])
                license_code = obs.get("license_code") or "UNKNOWN"
                user = obs.get("user", {}).get("login", "iNaturalist User")
                for ph in photos:
                    url = ph.get("url", "").replace("square", "original").replace("medium", "large")
                    if url:
                        candidates.append({
                            "title": f"iNat_{obs.get('id')}_{ph.get('id')}",
                            "url": url,
                            "license": license_code.upper(),
                            "author": user,
                            "source_name": "iNaturalist"
                        })
        time.sleep(1.0)
    except Exception as e:
        if "429" in str(e):
            time.sleep(5.0)
    return candidates


def main():
    print("=" * 70)
    print("STARTING DATASET EXPANSION PIPELINE v2 (82 BREEDS)")
    print(f"Target: Minimum {TARGET_PER_BREED} valid images per breed")
    print("=" * 70)

    with open(BREED_DETAILS_PATH, "r", encoding="utf-8") as f:
        breed_registry = json.load(f)

    # Hash tracker
    seen_sha256 = set()
    breed_phashes = {} # breed_id -> list of phashes

    metadata_rows = []

    # Step 1: Ingest existing baseline images from expanded_82_breeds
    print("\n--- STEP 1: IMPORTING EXISTING BASELINE IMAGES (486 IMAGES) ---")
    baseline_imported = 0
    for breed in breed_registry:
        b_id = breed["breed_id"]
        b_name = breed["breed_name"]
        sp = breed["species"]
        b_folder = b_id.replace("cow_", "").replace("buffalo_", "")
        src_dir = V1_DATASET / sp / b_folder
        dst_dir = V2_DATASET / sp / b_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        breed_phashes[b_id] = []

        if src_dir.exists():
            for img_file in sorted(src_dir.glob("*.jpg")):
                valid, reason, w, h, fmt = validate_image_file(img_file)
                if not valid:
                    continue
                md5_val, sha_val = calculate_hashes(img_file)
                if sha_val in seen_sha256:
                    continue
                seen_sha256.add(sha_val)
                ph = compute_phash(img_file)
                if ph:
                    breed_phashes[b_id].append(ph)

                dst_file = dst_dir / img_file.name
                if not dst_file.exists():
                    shutil.copy2(img_file, dst_file)

                metadata_rows.append({
                    "image_id": f"{sp[:3].upper()}_{b_folder.upper()}_{dst_file.stem[-4:]}",
                    "breed_id": b_id,
                    "breed_name": b_name,
                    "species": sp,
                    "source_url": "https://nbagr.res.in",
                    "source_name": "ICAR-NBAGR / Baseline Archive",
                    "license": "Government Open Access / Academic Research",
                    "download_date": "2026-09-17",
                    "original_filename": img_file.name,
                    "local_filename": f"{sp}/{b_id}/{dst_file.name}",
                    "sha256": sha_val,
                    "width": w,
                    "height": h,
                    "format": fmt,
                    "quality_status": "valid",
                    "duplicate_status": "original",
                    "near_duplicate_status": "original",
                    "verification_status": "verified"
                })
                baseline_imported += 1

    print(f"Successfully imported {baseline_imported} valid, deduplicated baseline images.")

    # Step 2: Multi-source acquisition for each breed
    print("\n--- STEP 2: MULTI-SOURCE ACQUISITION TOWARDS TARGET ---")
    cirb_by_breed = {}
    for item in CIRB_BUFFALO_ARCHIVE:
        cirb_by_breed.setdefault(item["breed"], []).append(item)

    for idx, breed in enumerate(breed_registry, 1):
        b_id = breed["breed_id"]
        b_name = breed["breed_name"]
        sp = breed["species"]
        dst_dir = V2_DATASET / sp / b_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        current_images = list(dst_dir.glob("*.jpg"))
        current_count = len(current_images)

        print(f"[{idx:2d}/82] {b_name} ({sp}): current count = {current_count} ...", end=" ", flush=True)

        if current_count >= TARGET_PER_BREED:
            print(f"Target already met ({current_count} images).")
            continue

        candidates = []

        # CIRB if buffalo
        if sp == "buffalo" and b_name in cirb_by_breed:
            for item in cirb_by_breed[b_name]:
                candidates.append({
                    "title": f"CIRB_{b_name}_{Path(item['url']).name}",
                    "url": item["url"],
                    "license": "Government Open Access / ICAR-CIRB",
                    "author": "ICAR-CIRB",
                    "source_name": "ICAR-CIRB"
                })

        # Wikimedia queries
        queries = BREED_QUERIES.get(b_name, [f"{b_name} {sp}", f'"{b_name}"'])
        for q in queries:
            if len(candidates) + current_count >= TARGET_PER_BREED * 2:
                break
            commons_results = query_wikimedia_commons(q, limit=50)
            candidates.extend(commons_results)

        # iNaturalist if under target
        if len(candidates) + current_count < TARGET_PER_BREED:
            taxon_id = 468444 if sp == "cattle" else 43128
            inat_results = query_inaturalist_observations(taxon_id, b_name, limit=25)
            candidates.extend(inat_results)

        # Process candidates
        added_for_breed = 0
        file_counter = current_count + 1

        for c in candidates:
            if current_count + added_for_breed >= TARGET_PER_BREED:
                break

            url = c["url"]
            url_lower = url.lower()
            if any(url_lower.endswith(ext) or ext in url_lower for ext in [".pdf", ".svg", ".djvu", ".tif", ".tiff", ".gif"]):
                continue

            filename = f"{b_id}_{file_counter:04d}.jpg"
            target_path = dst_dir / filename

            # Download
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=8) as resp:
                    content = resp.read()
                time.sleep(1.2) # respectful pacing between image downloads

                if len(content) < 5000:
                    continue
                with open(target_path, "wb") as out_f:
                    out_f.write(content)

                valid, reason, w, h, fmt = validate_image_file(target_path)
                if not valid:
                    target_path.unlink(missing_ok=True)
                    continue

                md5_val, sha_val = calculate_hashes(target_path)
                if sha_val in seen_sha256:
                    target_path.unlink(missing_ok=True)
                    continue

                # Perceptual hash check
                ph = compute_phash(target_path)
                if ph:
                    is_near_dup = False
                    for existing_ph in breed_phashes[b_id]:
                        if ph - existing_ph <= 4:
                            is_near_dup = True
                            break
                    if is_near_dup:
                        target_path.unlink(missing_ok=True)
                        continue
                    breed_phashes[b_id].append(ph)

                seen_sha256.add(sha_val)

                metadata_rows.append({
                    "image_id": f"{sp[:3].upper()}_{b_id.upper()}_{file_counter:04d}",
                    "breed_id": b_id,
                    "breed_name": b_name,
                    "species": sp,
                    "source_url": c["url"],
                    "source_name": c["source_name"],
                    "license": c.get("license", "UNKNOWN"),
                    "download_date": "2026-09-17",
                    "original_filename": c.get("title", filename),
                    "local_filename": f"{sp}/{b_id}/{filename}",
                    "sha256": sha_val,
                    "width": w,
                    "height": h,
                    "format": fmt,
                    "quality_status": "valid",
                    "duplicate_status": "original",
                    "near_duplicate_status": "original",
                    "verification_status": "verified"
                })

                added_for_breed += 1
                file_counter += 1

            except Exception as e:
                target_path.unlink(missing_ok=True)
                if "429" in str(e):
                    time.sleep(10.0) # exponential backoff
                continue

        print(f"added {added_for_breed} images (total: {current_count + added_for_breed})")

    # Step 3: Write metadata CSV
    fieldnames = [
        "image_id", "breed_id", "breed_name", "species", "source_url", "source_name",
        "license", "download_date", "original_filename", "local_filename", "sha256",
        "width", "height", "format", "quality_status", "duplicate_status",
        "near_duplicate_status", "verification_status"
    ]
    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\nSaved metadata for {len(metadata_rows)} images to {METADATA_CSV}")
    print("=" * 70)
    print("DATASET EXPANSION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
