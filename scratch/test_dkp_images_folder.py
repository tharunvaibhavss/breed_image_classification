import urllib.request
import json
import time

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

missing = [
    "Khariar", "Lakhimi", "Dagri", "Thutho", "Himachali Pahari",
    "Purnea", "Kathani", "Masilum", "Rohilkhandi", "Mahakaushali",
    "Umarda", "Melghati", "Gomanchali", "Mewati", "Shweta Kapila",
    "Konkan Kapila", "Poda Thurpu"
]

found = {}

for b in missing:
    b_title = b.replace(' ', '_')
    b_upper = b.upper().replace(' ', '_')
    variants = [
        f"{b_title}_Cow.jpg",
        f"{b_title}_Bull.jpg",
        f"{b_title}.jpg",
        f"{b_title}_cow.jpg",
        f"{b_title}_bull.jpg",
        f"{b_upper}_COW.jpg",
        f"{b_upper}_BULL.jpg"
    ]
    # Special cases
    if b == "Himachali Pahari":
        variants.extend(["Himachali_Pahari_Cow.jpg", "Himachali_cow.jpg", "Pahari_Cow.jpg"])

    for v in variants:
        url = f"https://www.dairyknowledge.in/sites/default/files/images/{v}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=4) as resp:
                if resp.status == 200:
                    print(f"FOUND DKP-images: {b} -> {url}")
                    found.setdefault(b, []).append(url)
        except Exception:
            pass
        time.sleep(0.04)

print(f"\nResolved in /sites/default/files/images/: {len(found)}")
