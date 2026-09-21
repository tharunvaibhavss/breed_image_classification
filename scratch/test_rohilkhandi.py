import urllib.request
from pathlib import Path
from PIL import Image

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    'Referer': 'https://www.researchgate.net/'
}

url = "https://www.researchgate.net/profile/Vallabhaneni-Srikanth/publication/404631474/figure/fig2/AS:11431282110513826@1778267439827/Rohilkhandi-cattle-with-its-owner_Q320.jpg"
dest = Path("dataset/raw/cattle/rohilkhandi/rohilkhandi_0001.jpg")
dest.parent.mkdir(parents=True, exist_ok=True)

try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        content = resp.read()
        with open(dest, "wb") as f:
            f.write(content)
        with Image.open(dest) as im:
            print("Downloaded Rohilkhandi:", im.size, len(content))
except Exception as e:
    print("Failed Rohilkhandi:", e)
