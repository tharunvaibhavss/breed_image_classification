import urllib.request, ssl
from PIL import Image
import io, os

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}

# Test variations
queries = [
    ("cattle", "bachaur", ["bachaur-bull.jpg", "bachaur_bull.jpg", "bachaur.png", "bachaur_cow.png", "bachaur_bull.png"]),
    ("cattle", "kosali", ["kosali-bull.jpg", "kosali_bull.jpg", "kosali_cow.jpg", "kosali.png", "kosali_cow.png", "kosali_bull.png"]),
    ("cattle", "ghumusari", ["ghumusari-bull.jpg", "ghumusari_bull.jpg", "ghumusari_cow.jpg", "ghumusari.jpg", "ghumsur_cow.jpg", "ghumsur_bull.jpg"]),
    ("cattle", "khariar", ["khariar-bull.jpg", "khariar_bull.jpg", "khariar.png", "khariar_cow.png", "khariar_bull.png", "khariar.jpg"]),
]

for sp, folder, varnames in queries:
    existing_fp = os.path.join("dataset", "raw", sp, folder, f"{folder}_0001.jpg")
    if not os.path.exists(existing_fp):
        existing_fp = os.path.join("dataset", "raw", sp, folder, f"{folder}_0001.png")
    with open(existing_fp, "rb") as fd:
        existing_data = fd.read()
    
    for v in varnames:
        url = f"https://www.dairyknowledge.in/dkp/sites/default/files/styles/medium_large/public/{v}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=4) as r:
                data = r.read()
                if data != existing_data and len(data) > 1000:
                    im = Image.open(io.BytesIO(data))
                    print(f"FOUND NEW {folder}: {v} ({im.size}, {len(data)} bytes)")
                    dest = os.path.join("dataset", "raw", sp, folder, f"{folder}_0002.jpg")
                    with open(dest, "wb") as f:
                        f.write(data)
                    break
        except Exception:
            pass
