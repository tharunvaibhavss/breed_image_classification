import urllib.request
import re
from pathlib import Path

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# 1. Download Mewati and Khariar
raw_root = Path("dataset/raw")
for sp, folder, url, fname in [
    ("cattle", "mewati", "https://www.dairyknowledge.in/dkp/sites/default/files/styles/medium_large/public/mewati-cow.jpg", "mewati_0001.jpg"),
    ("cattle", "khariar", "https://www.dairyknowledge.in/dkp/sites/default/files/styles/medium_large/public/khariar-cow.jpg", "khariar_0001.jpg")
]:
    dest = raw_root / sp / folder / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read()
            with open(dest, "wb") as f:
                f.write(content)
            print(f"Downloaded {sp}/{folder}/{fname}: {len(content)} bytes")
    except Exception as e:
        print(f"Failed {url}: {e}")

# 2. Check ccari.res.in for Gomanchali
try:
    req = urllib.request.Request("https://ccari.res.in/", headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        matches = [m for m in re.findall(r'<img[^>]+src=[\'"](.*?)[\'"]', html) if any(k in m.lower() for k in ['gomanchali', 'buffalo', 'breed'])]
        print("CCARI matches:", matches)
except Exception as e:
    print("CCARI error:", e)
