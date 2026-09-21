import urllib.request
import json
import time

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; AI Breed Recognition)'}

with open('dataset/metadata/breed_details.json', 'r', encoding='utf-8') as f:
    breeds = json.load(f)

# The 25 missing breeds
missing = [
    "Bachaur", "Kherigarh", "Khillar", "Mewati", "Ghumusari",
    "Binjharpuri", "Khariar", "Kosali", "Malnad Gidda", "Lakhimi",
    "Konkan Kapila", "Poda Thurpu", "Dagri", "Thutho", "Shweta Kapila",
    "Himachali Pahari", "Purnea", "Kathani", "Masilum", "Rohilkhandi",
    "Mahakaushali", "Umarda", "Marathwadi", "Melghati", "Gomanchali"
]

found_urls = {}

for b_name in missing:
    clean = b_name.lower().replace(' ', '_').replace('-', '_').split('(')[0].strip('_')
    variants = [
        f"{clean}.png",
        f"{clean}_cow.png",
        f"{clean}_bull.png",
        f"{clean}_female.png",
        f"{clean}_male.png",
        f"{clean}.jpg",
        f"{clean}_cow.jpg",
        f"{clean}_bull.jpg",
        f"{clean}_female.jpg",
        f"{clean}_male.jpg",
        f"{clean}_0.png",
        f"{clean}_0.jpg"
    ]
    if clean == "malnad_gidda":
        variants.extend(["malnad_gidda.png", "malnadgidda.png", "malnadgidda.jpg", "malnad_gudda.png"])
    elif clean == "konkan_kapila":
        variants.extend(["konkankapila.jpg", "konkankapila.png", "konkan_kapila.png", "kapila.png", "kapila.jpg"])
    elif clean == "shweta_kapila":
        variants.extend(["shwetakapila.png", "shwetakapila.jpg", "shweta_kapila.png", "shweta_kapila.jpg"])
    elif clean == "himachali_pahari":
        variants.extend(["himachali_pahari.png", "himachalipahari.png", "himachalipahari.jpg", "pahari.png", "pahari.jpg"])
    elif clean == "poda_thurpu":
        variants.extend(["podathurpu.png", "podathurpu.jpg", "poda_thurpu.png"])

    for v in variants:
        url = f"https://www.dairyknowledge.in/dkp/sites/default/files/styles/medium_large/public/{v}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=4) as resp:
                if resp.status == 200:
                    print(f"FOUND DKP: {b_name} -> {url}")
                    found_urls.setdefault(b_name, []).append(url)
        except Exception:
            pass
        time.sleep(0.04)

print(f"\nResolved {len(found_urls)} / {len(missing)} missing breeds on DKP!")
