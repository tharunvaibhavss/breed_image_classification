import urllib.request

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; AI Breed Recognition)'}

test_breeds = [
    "kosali", "bachaur", "kherigarh", "mewati", "marathwadi",
    "ghumusari", "binjharpuri", "khariar", "lakhimi", "dagri",
    "thutho", "purnea", "kathani", "malnad_gidda", "khillar"
]

for b in test_breeds:
    for suffix in [f"{b}.jpg", f"{b}.png", f"{b}_0.jpg", f"{b}_bull.jpg", f"{b}_cow.jpg"]:
        url = f"https://www.dairyknowledge.in/dkp/sites/default/files/styles/medium_large/public/{suffix}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                print(f"FOUND: {url} -> {resp.status}")
                break
        except Exception:
            pass
