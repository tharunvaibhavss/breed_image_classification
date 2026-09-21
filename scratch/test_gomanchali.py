import urllib.request
from PIL import Image

headers = {'User-Agent': 'Mozilla/5.0'}
urls = [
    "https://ccari.res.in/Gomanchali210726-4.JPG",
    "https://ccari.res.in/Gomanchali210726-4_tmb.JPG",
    "https://ccari.res.in/Gomanchali210726-1.JPG",
    "https://ccari.res.in/Gomanchali210726-2.JPG"
]

for u in urls:
    try:
        req = urllib.request.Request(u, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            print(f"FOUND: {u} -> len {len(data)}")
            with open("test_gom.jpg", "wb") as f:
                f.write(data)
            with Image.open("test_gom.jpg") as im:
                print("Size:", im.size)
    except Exception as e:
        print(f"{u} -> {e}")
