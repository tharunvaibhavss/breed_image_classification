import urllib.request
import json
import time

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; AI Breed Recognition)'}

with open('dataset/metadata/breed_details.json', 'r', encoding='utf-8') as f:
    breeds = json.load(f)

dkp_found = {}

for b in breeds:
    b_name = b['breed_name'].lower().replace(' ', '_').replace('-', '_').split('(')[0].strip('_')
    variants = [
        f"{b_name}.jpg",
        f"{b_name}_cow.jpg",
        f"{b_name}_bull.jpg",
        f"{b_name}_female.jpg",
        f"{b_name}_male.jpg",
        f"{b_name}_0.jpg"
    ]
    # Also handle some special spellings
    if b_name == "malnad_gidda":
        variants.extend(["malnad_gudda.jpg", "malnadgidda.jpg"])
    elif b_name == "poda_thurpu":
        variants.extend(["podathurpu.jpg", "podathurpu_bull.jpg"])
    elif b_name == "red_kandhari":
        variants.extend(["kandhari.jpg", "kandhari_bull.jpg", "redkandhari.jpg"])
    elif b_name == "krishna_valley":
        variants.extend(["krishnavalley.jpg", "krishna_valley_bull.jpg"])

    for v in variants:
        url = f"https://www.dairyknowledge.in/dkp/sites/default/files/styles/medium_large/public/{v}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"DKP Match: {b['breed_name']} -> {url}")
                    dkp_found.setdefault(b['breed_name'], []).append(url)
        except Exception:
            pass
        time.sleep(0.05)

with open('scratch/dkp_matches.json', 'w', encoding='utf-8') as f:
    json.dump(dkp_found, f, indent=2)

print(f"\nTotal breeds found on DKP: {len(dkp_found)}")
