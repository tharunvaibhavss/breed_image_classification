import urllib.request, ssl
from PIL import Image
import io, os

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

targets = [
    ("cattle", "mewati", "mewati-cow.jpg"),
    ("cattle", "bachaur", "bachaur-cow.jpg"),
    ("cattle", "belahi", "belahi-cow.jpg"),
    ("cattle", "kosali", "kosali-cow.jpg"),
    ("cattle", "khariar", "khariar-cow.jpg"),
    ("cattle", "ghumusari", "ghumusari-cow.jpg"),
    ("cattle", "koppal", "koppal-cow.jpg"),
    ("cattle", "umarda", "umarda-cow.jpg"),
    ("cattle", "mewati", "mewati.jpg"),
    ("cattle", "bachaur", "bachaur.jpg"),
    ("cattle", "belahi", "belahi.jpg"),
    ("cattle", "kosali", "kosali.jpg"),
    ("cattle", "khariar", "khariar.jpg"),
    ("cattle", "ghumusari", "ghumusari.jpg"),
]

for sp, folder, imgname in targets:
    url = f"https://www.dairyknowledge.in/dkp/sites/default/files/styles/medium_large/public/{imgname}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=5) as r:
            data = r.read()
            im = Image.open(io.BytesIO(data))
            print(f"SUCCESS: {folder} -> {imgname} ({im.size}, {len(data)} bytes)")
            dest = os.path.join("dataset", "raw", sp, folder, f"{folder}_0002.jpg")
            with open(dest, "wb") as f:
                f.write(data)
            print(f"Saved {dest}")
    except Exception as e:
        pass
