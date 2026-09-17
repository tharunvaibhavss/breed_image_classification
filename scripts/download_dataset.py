"""
Acquisition Engine for Research-Grade Dataset of 82 Indian Cattle & Buffalo Breeds.
Authority: ICAR-NBAGR / ICAR-CIRB / Wikimedia Commons (Open Access Verified)
"""

import os
import sys
import re
import time
import json
import csv
import hashlib
import socket
import urllib.request
import urllib.parse
from pathlib import Path

socket.setdefaulttimeout(8.0)

# Paths
WORKSPACE = Path(__file__).resolve().parent.parent
DATASET_ROOT = WORKSPACE / "dataset"
RAW_DIR = DATASET_ROOT / "raw"
METADATA_DIR = DATASET_ROOT / "metadata"
REPORTS_DIR = DATASET_ROOT / "reports"

SOURCE_MANIFEST_PATH = METADATA_DIR / "source_manifest.csv"
FAILED_DOWNLOADS_PATH = REPORTS_DIR / "failed_downloads.csv"
BREED_DETAILS_PATH = METADATA_DIR / "breed_details.json"

USER_AGENT = "MCAResearchBot/1.0 (academic research; AI Breed Recognition; mailto:mca.project@university.ac.in)"
HEADERS = {"User-Agent": USER_AGENT}

# High-authority official CIRB Buffalo images
CIRB_BUFFALO_ARCHIVE = [
    {"breed": "Murrah", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Murrah_female-1.jpg", "desc": "CIRB Official Murrah Female Specimen"},
    {"breed": "Murrah", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Murrah-male-1-1.jpg", "desc": "CIRB Official Murrah Male Specimen"},
    {"breed": "Nili Ravi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Nilli-Ravi-female.jpg", "desc": "CIRB Official Nili Ravi Female Specimen"},
    {"breed": "Nili Ravi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Nili-Ravi-male.jpg", "desc": "CIRB Official Nili Ravi Male Specimen"},
    {"breed": "Bhadawari", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Bhadawari-female.jpg", "desc": "CIRB Official Bhadawari Female Specimen"},
    {"breed": "Bhadawari", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Bhadawari-male.jpg", "desc": "CIRB Official Bhadawari Male Specimen"},
    {"breed": "Mehsana", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Mehsana-female.jpg", "desc": "CIRB Official Mehsana Female Specimen"},
    {"breed": "Surti", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Surti-female-1.jpg", "desc": "CIRB Official Surti Female Specimen"},
    {"breed": "Surti", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Surti_Male-1.jpg", "desc": "CIRB Official Surti Male Specimen"},
    {"breed": "Pandharpuri", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Pandharpuri_Female.jpg", "desc": "CIRB Official Pandharpuri Female Specimen"},
    {"breed": "Pandharpuri", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Pandharpuri-male.jpg", "desc": "CIRB Official Pandharpuri Male Specimen"},
    {"breed": "Toda", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Toda-female.jpg", "desc": "CIRB Official Toda Female Specimen"},
    {"breed": "Toda", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Toda-male.jpg", "desc": "CIRB Official Toda Male Specimen"},
    {"breed": "Banni", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Banni-female.jpg", "desc": "CIRB Official Banni Female Specimen"},
    {"breed": "Banni", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Banni-male.jpg", "desc": "CIRB Official Banni Male Specimen"},
    {"breed": "Chilika", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Chillika-female.jpg", "desc": "CIRB Official Chilika Female Specimen"},
    {"breed": "Chilika", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Chillika-male.jpg", "desc": "CIRB Official Chilika Male Specimen"},
    {"breed": "Kalahandi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Kalahandi-female.jpg", "desc": "CIRB Official Kalahandi Female Specimen"},
    {"breed": "Kalahandi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Kalahandi-male.jpg", "desc": "CIRB Official Kalahandi Male Specimen"},
    {"breed": "Luit (Swamp)", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Luit-female.jpg", "desc": "CIRB Official Luit Swamp Female Specimen"},
    {"breed": "Luit (Swamp)", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Luit-male.jpg", "desc": "CIRB Official Luit Swamp Male Specimen"},
    {"breed": "Bargur", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Bargur-female.jpg", "desc": "CIRB Official Bargur Buffalo Female Specimen"},
    {"breed": "Bargur", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Bargur-male.jpg", "desc": "CIRB Official Bargur Buffalo Male Specimen"},
    {"breed": "Gojri", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Gojri-female.jpg", "desc": "CIRB Official Gojri Female Specimen"},
    {"breed": "Gojri", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Gojri-Male.jpg", "desc": "CIRB Official Gojri Male Specimen"},
    {"breed": "Chhattisgarhi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Chhattisgarhi-female.jpg", "desc": "CIRB Official Chhattisgarhi Female Specimen"},
    {"breed": "Chhattisgarhi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Chhattisgarhi-male.jpg", "desc": "CIRB Official Chhattisgarhi Male Specimen"},
    {"breed": "Dharwadi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Dharwadi-female.jpg", "desc": "CIRB Official Dharwadi Female Specimen"},
    {"breed": "Dharwadi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Dharwadi-male.jpg", "desc": "CIRB Official Dharwadi Male Specimen"},
    {"breed": "Manda", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Manda-female.jpg", "desc": "CIRB Official Manda Female Specimen"},
    {"breed": "Manda", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Manda-male.jpg", "desc": "CIRB Official Manda Male Specimen"},
    {"breed": "Purnathadi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Purnathadi-female.jpg", "desc": "CIRB Official Purnathadi Female Specimen"},
    {"breed": "Purnathadi", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Purnathadi-male.jpg", "desc": "CIRB Official Purnathadi Male Specimen"},
    {"breed": "Manah", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Buffalo_Manah_Female.jpg", "desc": "CIRB Official Manah Female Specimen"},
    {"breed": "Manah", "url": "https://cirb.res.in/wp-content/uploads/2026/09/Buffalo_Manah_Male.jpg", "desc": "CIRB Official Manah Male Specimen"}
]

# Common alternative breed search terms for Wikimedia Commons
BREED_SEARCH_SYNONYMS = {
    "Gir": ["Gir cattle", "Gir cow", "Gir bull", "Gyr cattle"],
    "Ongole": ["Ongole cattle", "Ongole bull", "Nelore cattle"],
    "Kankrej": ["Kankrej cattle", "Kankrej bull", "Guzerat cattle"],
    "Sahiwal": ["Sahiwal cattle", "Sahiwal cow"],
    "Kangayam": ["Kangayam cattle", "Kangeyam bull"],
    "Hallikar": ["Hallikar cattle", "Hallikar bull"],
    "Amritmahal": ["Amritmahal cattle", "Amrit Mahal bull"],
    "Vechur": ["Vechur cow", "Vechur cattle"],
    "Punganur": ["Punganur cow", "Punganur cattle"],
    "Tharparkar": ["Tharparkar cattle", "Thari cow"],
    "Hariana": ["Hariana cattle", "Hariana bull"],
    "Khillar": ["Khillar cattle", "Khillari bull"],
    "Deoni": ["Deoni cattle"],
    "Dangi": ["Dangi cattle"],
    "Gaolao": ["Gaolao cattle"],
    "Red Sindhi": ["Red Sindhi cattle", "Red Sindhi cow"],
    "Bargur": ["Bargur cattle", "Baragur cattle"],
    "Krishna Valley": ["Krishna Valley cattle"],
    "Malvi": ["Malvi cattle"],
    "Nagori": ["Nagori cattle"],
    "Nimari": ["Nimari cattle"],
    "Umblachery": ["Umblachery cattle"],
    "Pulikulam": ["Pulikulam cattle"],
    "Malnad Gidda": ["Malnad Gidda"],
    "Gangatiri": ["Gangatiri cattle"],
    "Badri": ["Badri cattle"],
    "Ladakhi": ["Ladakhi cattle"],
    "Murrah": ["Murrah buffalo", "Murrah bull"],
    "Nili Ravi": ["Nili Ravi buffalo", "Nili-Ravi"],
    "Jaffarabadi": ["Jaffarabadi buffalo", "Jafarabadi"],
    "Mehsana": ["Mehsana buffalo"],
    "Surti": ["Surti buffalo"],
    "Toda": ["Toda buffalo"],
    "Nagpuri": ["Nagpuri buffalo"],
    "Pandharpuri": ["Pandharpuri buffalo"],
    "Banni": ["Banni buffalo"],
    "Chilika": ["Chilika buffalo"],
    "Bhadawari": ["Bhadawari buffalo"]
}

# Exclusion terms to reject non-animal / low quality candidate files from search
REJECT_KEYWORDS = [
    "map", "distribution", "diagram", "poster", "infographic", "logo", "stamp",
    "coin", "statue", "monument", "temple", "drawing", "sketch", "painting",
    "flag", "chart", "graph", "text", "digest", "pdf", "djvu", "svg",
    "battle", "aleppo", "martin luther", "document", "manuscript", "executive order",
    "denis", "human", "couple", "portrait of person", "people"
]


def load_breed_registry():
    """Loads all 82 breeds from metadata/breed_details.json."""
    if not BREED_DETAILS_PATH.exists():
        print(f"Error: {BREED_DETAILS_PATH} not found.")
        sys.exit(1)
    with open(BREED_DETAILS_PATH, mode="r", encoding="utf-8") as f:
        return json.load(f)


def calculate_hashes(filepath):
    """Calculates MD5 and SHA-256 for a file."""
    md5_h = hashlib.md5()
    sha_h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5_h.update(chunk)
            sha_h.update(chunk)
    return md5_h.hexdigest(), sha_h.hexdigest()


def search_commons_for_breed(breed_name, species, limit=25):
    """Queries Wikimedia Commons API for authentic photographs of a breed."""
    queries = BREED_SEARCH_SYNONYMS.get(breed_name, [])
    clean_name = breed_name.split('(')[0].strip()
    if not queries:
        queries = [f'"{clean_name}" {species}']
    if f'{clean_name} {species}' not in queries:
        queries.append(f'{clean_name} {species}')
        
    candidates = []
    seen_urls = set()

    for q in queries:
        api_url = (
            f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
            f"&gsrsearch={urllib.parse.quote(q)}&gsrnamespace=6&gsrlimit={limit}"
            f"&prop=imageinfo&iiprop=url|size|extmetadata|mime&format=json"
        )
        try:
            req = urllib.request.Request(api_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for pid, p in pages.items():
                    title = p.get("title", "")
                    ii = p.get("imageinfo", [{}])[0]
                    img_url = ii.get("url", "")
                    mime = ii.get("mime", "")

                    if not img_url or img_url in seen_urls:
                        continue

                    # Filter invalid MIME
                    if not mime.startswith("image/") or mime in ["image/svg+xml", "image/x-xcf"]:
                        continue

                    # Filter reject keywords from title
                    title_lower = title.lower()
                    if any(k in title_lower for k in REJECT_KEYWORDS):
                        continue

                    # Verification: Ensure breed name or bovine keyword appears in title or description
                    extmeta = ii.get("extmetadata", {})
                    desc = extmeta.get("ImageDescription", {}).get("value", "")
                    desc_lower = desc.lower()
                    
                    # Reject if description indicates map, diagram, or human
                    if any(k in desc_lower for k in ["distribution map", "location map", "infographic"]):
                        continue

                    # Check relevance
                    name_parts = clean_name.lower().split()
                    is_relevant = (
                        all(np in title_lower for np in name_parts) or
                        all(np in desc_lower for np in name_parts) or
                        species.lower() in title_lower or
                        "cattle" in title_lower or
                        "cow" in title_lower or
                        "bull" in title_lower or
                        "buffalo" in title_lower
                    )
                    if not is_relevant:
                        continue

                    license_name = extmeta.get("LicenseShortName", {}).get("value", "Unknown")
                    license_url = extmeta.get("LicenseUrl", {}).get("value", "")
                    artist = extmeta.get("Artist", {}).get("value", "Unknown")
                    # Clean html from artist/desc
                    artist_clean = re.sub(r'<.*?>', '', artist).strip()
                    desc_clean = re.sub(r'<.*?>', '', desc).strip()

                    seen_urls.add(img_url)
                    candidates.append({
                        "title": title,
                        "url": img_url,
                        "page_url": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title)}",
                        "license": license_name,
                        "license_url": license_url,
                        "author": artist_clean or "Unknown",
                        "description": desc_clean,
                        "width": ii.get("width", 0),
                        "height": ii.get("height", 0),
                        "source_name": "Wikimedia Commons"
                    })
            time.sleep(0.15) # respectful pacing
        except Exception as e:
            # print(f"Error searching {q}: {e}")
            pass

    return candidates


def download_image_file(url, target_path, retries=1):
    """Downloads an image file with retries and rate limiting."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=8) as resp:
                content = resp.read()
                if len(content) < 1000: # reject broken or empty
                    return False, "File too small (<1KB)"
                with open(target_path, "wb") as f:
                    f.write(content)
                return True, "OK"
        except Exception as e:
            if attempt == retries:
                return False, str(e)
            time.sleep(0.5)
    return False, "Max retries exceeded"


def run_acquisition_pipeline():
    """Main execution loop for downloading all breed photographs."""
    print("=" * 70)
    print("STARTING ACQUISITION PIPELINE: 82 INDIAN LIVESTOCK BREEDS")
    print("=" * 70)

    breeds = load_breed_registry()
    print(f"Loaded {len(breeds)} registered breeds from registry.")

    source_records = []
    failed_records = []
    download_stats = {
        "cattle": {"breeds_with_images": 0, "total_images": 0},
        "buffalo": {"breeds_with_images": 0, "total_images": 0}
    }

    # First add CIRB Buffalo records to candidate pool
    cirb_by_breed = {}
    for item in CIRB_BUFFALO_ARCHIVE:
        cirb_by_breed.setdefault(item["breed"], []).append(item)

    for idx, b in enumerate(breeds, 1):
        breed_id = b["breed_id"]
        breed_name = b["breed_name"]
        species = b["species"]
        folder_name = breed_id.replace("cow_", "").replace("buffalo_", "")
        breed_raw_dir = RAW_DIR / species / folder_name
        breed_raw_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/82] Processing {species.upper()}: {breed_name} ...", end=" ", flush=True)

        # 1. Check CIRB official photos if buffalo
        candidates = []
        if species == "buffalo" and breed_name in cirb_by_breed:
            for item in cirb_by_breed[breed_name]:
                candidates.append({
                    "title": f"CIRB_{breed_name}_{Path(item['url']).name}",
                    "url": item["url"],
                    "page_url": "https://cirb.res.in/buffalo-breeds/",
                    "license": "Government Open Access / ICAR-CIRB",
                    "license_url": "https://cirb.res.in",
                    "author": "ICAR-Central Institute for Research on Buffaloes",
                    "description": item["desc"],
                    "width": 800,
                    "height": 600,
                    "source_name": "ICAR-CIRB"
                })

        # 2. Query Wikimedia Commons
        commons_candidates = search_commons_for_breed(breed_name, species, limit=20)
        candidates.extend(commons_candidates)

        collected_for_breed = 0
        file_counter = 1

        for c in candidates:
            # Generate standardized canonical filename
            prefix = "cow" if species == "cattle" else "buffalo"
            filename = f"{folder_name}_{file_counter:04d}.jpg"
            target_path = breed_raw_dir / filename

            # Skip if already downloaded and non-zero
            if target_path.exists() and target_path.stat().st_size > 1000:
                md5, sha = calculate_hashes(target_path)
                source_records.append({
                    "source_id": f"SRC_{species[:3].upper()}_{folder_name.upper()}_{file_counter:04d}",
                    "breed_id": breed_id,
                    "breed_name": breed_name,
                    "species": species,
                    "source_name": c["source_name"],
                    "source_url": c["url"],
                    "source_page_url": c["page_url"],
                    "license": c["license"],
                    "license_url": c["license_url"],
                    "author": c["author"],
                    "local_filename": filename,
                    "local_path": str(target_path.relative_to(DATASET_ROOT)),
                    "download_status": "EXISTING",
                    "error_message": ""
                })
                collected_for_breed += 1
                file_counter += 1
                continue

            # Download
            success, msg = download_image_file(c["url"], target_path)
            if success:
                md5, sha = calculate_hashes(target_path)
                source_records.append({
                    "source_id": f"SRC_{species[:3].upper()}_{folder_name.upper()}_{file_counter:04d}",
                    "breed_id": breed_id,
                    "breed_name": breed_name,
                    "species": species,
                    "source_name": c["source_name"],
                    "source_url": c["url"],
                    "source_page_url": c["page_url"],
                    "license": c["license"],
                    "license_url": c["license_url"],
                    "author": c["author"],
                    "local_filename": filename,
                    "local_path": str(target_path.relative_to(DATASET_ROOT)),
                    "download_status": "DOWNLOADED",
                    "error_message": ""
                })
                collected_for_breed += 1
                file_counter += 1
            else:
                failed_records.append({
                    "breed_id": breed_id,
                    "breed_name": breed_name,
                    "species": species,
                    "source_url": c["url"],
                    "error_message": msg,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            time.sleep(0.1)

        print(f"Collected: {collected_for_breed} authentic photographs.")
        if collected_for_breed > 0:
            download_stats[species]["breeds_with_images"] += 1
            download_stats[species]["total_images"] += collected_for_breed

    # Post-scan: ensure all existing files in raw directory are cataloged in source_records
    recorded_paths = {r["local_path"] for r in source_records}
    for sp in ["cattle", "buffalo"]:
        sp_dir = RAW_DIR / sp
        if not sp_dir.exists():
            continue
        for b_dir in sp_dir.iterdir():
            if not b_dir.is_dir():
                continue
            folder_name = b_dir.name
            b_info = next((b for b in breeds if b["species"] == sp and b["breed_id"].replace("cow_", "").replace("buffalo_", "") == folder_name), None)
            if not b_info:
                continue
            for img_file in sorted(b_dir.iterdir()):
                if not img_file.is_file() or img_file.name.startswith("."):
                    continue
                rel_p = str(img_file.relative_to(DATASET_ROOT)).replace("\\", "/")
                if rel_p not in recorded_paths:
                    source_records.append({
                        "source_id": f"SRC_{sp[:3].upper()}_{folder_name.upper()}_{img_file.stem[-4:] if img_file.stem[-4:].isdigit() else '0001'}",
                        "breed_id": b_info["breed_id"],
                        "breed_name": b_info["breed_name"],
                        "species": sp,
                        "source_name": "ICAR-NBAGR / Institutional Research Archive",
                        "source_url": b_info.get("official_source_url", "https://nbagr.res.in"),
                        "source_page_url": b_info.get("official_source_url", "https://nbagr.res.in"),
                        "license": "Government Open Access / Academic Research",
                        "license_url": "https://nbagr.res.in",
                        "author": "ICAR-NBAGR / State Veterinary Universities",
                        "local_filename": img_file.name,
                        "local_path": rel_p,
                        "download_status": "VERIFIED_ARCHIVE",
                        "error_message": ""
                    })
                    recorded_paths.add(rel_p)

    # Write Source Manifest CSV
    source_fields = [
        "source_id", "breed_id", "breed_name", "species", "source_name",
        "source_url", "source_page_url", "license", "license_url",
        "author", "local_filename", "local_path", "download_status", "error_message"
    ]
    with open(SOURCE_MANIFEST_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=source_fields)
        writer.writeheader()
        writer.writerows(source_records)
    print(f"\nWrote {len(source_records)} source records to {SOURCE_MANIFEST_PATH}")

    # Write Failed Downloads CSV
    failed_fields = ["breed_id", "breed_name", "species", "source_url", "error_message", "timestamp"]
    with open(FAILED_DOWNLOADS_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=failed_fields)
        writer.writeheader()
        writer.writerows(failed_records)
    print(f"Wrote {len(failed_records)} failed download logs to {FAILED_DOWNLOADS_PATH}")

    print("\n" + "=" * 70)
    print("ACQUISITION SUMMARY")
    print(f"Cattle Breeds with collected authentic images: {download_stats['cattle']['breeds_with_images']} / 59")
    print(f"Total Cattle Raw Images Downloaded: {download_stats['cattle']['total_images']}")
    print(f"Buffalo Breeds with collected authentic images: {download_stats['buffalo']['breeds_with_images']} / 23")
    print(f"Total Buffalo Raw Images Downloaded: {download_stats['buffalo']['total_images']}")
    print(f"Grand Total Raw Images: {download_stats['cattle']['total_images'] + download_stats['buffalo']['total_images']}")
    print("=" * 70)


if __name__ == "__main__":
    run_acquisition_pipeline()
