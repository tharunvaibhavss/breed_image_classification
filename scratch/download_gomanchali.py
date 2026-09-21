import urllib.request
from pathlib import Path
from PIL import Image

headers = {'User-Agent': 'Mozilla/5.0'}
dest_dir = Path("dataset/raw/buffalo/gomanchali")
dest_dir.mkdir(parents=True, exist_ok=True)

urls = [
    ("https://ccari.res.in/Gomanchali210726-1.JPG", "gomanchali_0001.jpg"),
    ("https://ccari.res.in/Gomanchali210726-2.JPG", "gomanchali_0002.jpg"),
    ("https://ccari.res.in/Gomanchali210726-4.JPG", "gomanchali_0003.jpg")
]

for url, fname in urls:
    dest = dest_dir / fname
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        content = resp.read()
        with open(dest, "wb") as f:
            f.write(content)
        with Image.open(dest) as im:
            print(f"Downloaded Gomanchali: {fname} ({im.size}, {len(content)} bytes)")
