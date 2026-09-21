import urllib.request, ssl, os
from PIL import Image
import io

ctx = ssl._create_unverified_context()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

candidates = [
    # kherigarh -> Kerigar 02.JPG on Commons
    ("cattle", "kherigarh", "kherigarh_0002.jpg", "https://upload.wikimedia.org/wikipedia/commons/c/c5/Kerigar_02.JPG"),
    # mewati -> SaveIndianCows / Apni Kheti
    ("cattle", "mewati", "mewati_0002.jpg", "https://saveindiancows.org/wp-content/uploads/2016/07/Mewati.jpg"),
    # bachaur -> SaveIndianCows
    ("cattle", "bachaur", "bachaur_0002.jpg", "https://saveindiancows.org/wp-content/uploads/2016/07/Bachaur.jpg"),
    # kosali -> SaveIndianCows
    ("cattle", "kosali", "kosali_0002.jpg", "https://saveindiancows.org/wp-content/uploads/2016/07/Kosali.jpg"),
    # ghumusari -> SaveIndianCows
    ("cattle", "ghumusari", "ghumusari_0002.jpg", "https://saveindiancows.org/wp-content/uploads/2016/07/Ghumusari.jpg"),
    # khariar -> SaveIndianCows
    ("cattle", "khariar", "khariar_0002.jpg", "https://saveindiancows.org/wp-content/uploads/2016/07/Khariar.jpg"),
    # sanchori -> SaveIndianCows
    ("cattle", "sanchori", "sanchori_0002.jpg", "https://saveindiancows.org/wp-content/uploads/2016/07/Sanchori.jpg"),
]

for sp, folder, fname, url in candidates:
    dest = os.path.join("dataset", "raw", sp, folder, fname)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
            data = r.read()
            im = Image.open(io.BytesIO(data))
            print(f"SUCCESS {folder}: {im.size}, {im.format}, {len(data)} bytes")
            with open(dest, "wb") as f:
                f.write(data)
    except Exception as e:
        print(f"Failed {folder} ({url}): {e}")
