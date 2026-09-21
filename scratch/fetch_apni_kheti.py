import urllib.request, ssl, os
from PIL import Image
import io

ctx = ssl._create_unverified_context()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

apni_kheti = [
    ("cattle", "bachaur", "bachaur_0002.jpg", "https://www.apnikheti.com/upload/liveStock/4523idea99bachaur.jpg"),
    ("cattle", "kosali", "kosali_0002.jpg", "https://www.apnikheti.com/upload/liveStock/1407idea99kosali.jpg"),
    ("cattle", "ghumusari", "ghumusari_0002.jpg", "https://www.apnikheti.com/upload/liveStock/3909idea99ghumsuri.jpg"),
    ("cattle", "khariar", "khariar_0002.jpg", "https://www.apnikheti.com/upload/liveStock/7987idea99khariar.jpg"),
    ("cattle", "sanchori", "sanchori_0002.jpg", "https://www.apnikheti.com/upload/liveStock/7364idea99sanchori.jpg"),
]

for sp, folder, fname, url in apni_kheti:
    dest = os.path.join("dataset", "raw", sp, folder, fname)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
            data = r.read()
            im = Image.open(io.BytesIO(data))
            print(f"SUCCESS {folder}: {im.size}, {im.format}, {len(data)} bytes")
            with open(dest, "wb") as f:
                f.write(data)
    except Exception as e:
        print(f"Failed {folder}: {e}")
