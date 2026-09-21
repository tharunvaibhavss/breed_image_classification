import urllib.request, ssl
from PIL import Image
import io, os

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

test_names = [
    # bachaur
    ("cattle", "bachaur", ["bachaur-cow.jpg", "bachaur-bull.jpg", "bachaur_cow.jpg", "bachaur_bull.jpg", "bachaur.jpg", "bachore-cow.jpg", "bachore.jpg"]),
    # ghumusari
    ("cattle", "ghumusari", ["ghumusari-cow.jpg", "ghumusari.jpg", "ghumsur-cow.jpg", "ghumsur.jpg", "ghumsari.jpg", "ghumsari-cow.jpg"]),
    # koppal
    ("cattle", "koppal", ["koppal-cow.jpg", "koppal.jpg", "koppal_cow.jpg", "koppal-bull.jpg"]),
    # umarda
    ("cattle", "umarda", ["umarda-cow.jpg", "umarda.jpg", "umred.jpg", "umred-cow.jpg"]),
]

for sp, folder, variants in test_names:
    found = False
    for v in variants:
        url = f"https://www.dairyknowledge.in/dkp/sites/default/files/styles/medium_large/public/{v}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=4) as r:
                data = r.read()
                im = Image.open(io.BytesIO(data))
                print(f"FOUND DKP: {folder} -> {v} ({im.size})")
                dest = os.path.join("dataset", "raw", sp, folder, f"{folder}_0002.jpg")
                with open(dest, "wb") as f:
                    f.write(data)
                found = True
                break
        except Exception:
            pass
    if not found:
        print(f"Not found on DKP: {folder}")
