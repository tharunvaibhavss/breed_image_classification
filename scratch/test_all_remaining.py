import urllib.request
import json
import time

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

remaining = [
    "Khariar", "Kathani", "Masilum", "Rohilkhandi", "Mahakaushali",
    "Umarda", "Melghati", "Gomanchali", "Mewati"
]

found = {}

base_urls = [
    "https://www.dairyknowledge.in/sites/default/files/images/",
    "https://www.dairyknowledge.in/dkp/sites/default/files/styles/medium_large/public/",
    "https://www.dairyknowledge.in/sites/default/files/",
    "https://cirb.res.in/wp-content/uploads/2026/09/"
]

for b in remaining:
    b_title = b.replace(' ', '_')
    b_lower = b.lower().replace(' ', '_')
    names = [
        f"{b_title}_Cow.jpg", f"{b_title}_Bull.jpg", f"{b_title}.jpg",
        f"{b_lower}_cow.jpg", f"{b_lower}_bull.jpg", f"{b_lower}.jpg",
        f"{b_title}_Female.jpg", f"{b_title}_Male.jpg",
        f"{b_lower}_female.jpg", f"{b_lower}_male.jpg",
        f"{b_lower}_cow.png", f"{b_lower}_bull.png", f"{b_lower}.png",
        f"{b_title}.png", f"Buffalo_{b_title}_Female.jpg", f"Buffalo_{b_title}_Male.jpg"
    ]
    for base in base_urls:
        for n in names:
            url = f"{base}{n}"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=3) as resp:
                    if resp.status == 200:
                        print(f"FOUND: {b} -> {url}")
                        found.setdefault(b, []).append(url)
            except Exception:
                pass
            time.sleep(0.02)

print(f"\nRemaining resolved: {len(found)} / {len(remaining)}")
