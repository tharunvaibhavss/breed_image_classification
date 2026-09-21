import urllib.request
from pathlib import Path
from PIL import Image

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

downloads = [
    ("cattle", "shweta_kapila", "https://www.dairyknowledge.in/sites/default/files/images/Shweta_Kapila_Cow.jpg", "shweta_kapila_0001.jpg"),
    ("cattle", "shweta_kapila", "https://ccari.res.in/Shwetkapila2.jpg", "shweta_kapila_0002.jpg"),
    ("cattle", "konkan_kapila", "https://www.dairyknowledge.in/sites/default/files/images/Konkan_Kapila_Bull.jpg", "konkan_kapila_0001.jpg"),
    ("cattle", "konkan_kapila", "https://www.dairyknowledge.in/sites/default/files/images/Konkan_Kapila_Cow.jpg", "konkan_kapila_0002.jpg"),
    ("cattle", "poda_thurpu", "https://www.dairyknowledge.in/sites/default/files/images/Poda_Thurpu_Bull.jpg", "poda_thurpu_0001.jpg"),
    ("cattle", "poda_thurpu", "https://www.dairyknowledge.in/sites/default/files/images/Poda_Thurpu_Cow.jpg", "poda_thurpu_0002.jpg"),
    ("cattle", "lakhimi", "https://www.dairyknowledge.in/sites/default/files/images/Lakhimi_Cow.jpg", "lakhimi_0001.jpg"),
    ("cattle", "lakhimi", "https://www.dairyknowledge.in/sites/default/files/images/Lakhimi_Bull.jpg", "lakhimi_0002.jpg"),
    ("cattle", "dagri", "https://www.dairyknowledge.in/sites/default/files/images/Dagri_Cow.jpg", "dagri_0001.jpg"),
    ("cattle", "dagri", "https://www.dairyknowledge.in/sites/default/files/images/Dagri_Bull.jpg", "dagri_0002.jpg"),
    ("cattle", "thutho", "https://www.dairyknowledge.in/sites/default/files/images/Thutho_Cow.jpg", "thutho_0001.jpg"),
    ("cattle", "thutho", "https://www.dairyknowledge.in/sites/default/files/images/Thutho_Bull.jpg", "thutho_0002.jpg"),
    ("cattle", "himachali_pahari", "https://www.dairyknowledge.in/sites/default/files/images/Himachali_Pahari_Cow.jpg", "himachali_pahari_0001.jpg"),
    ("cattle", "himachali_pahari", "https://www.dairyknowledge.in/sites/default/files/images/Himachali_Pahari_Bull.jpg", "himachali_pahari_0002.jpg"),
    ("cattle", "purnea", "https://www.dairyknowledge.in/sites/default/files/images/Purnea_Cow.jpg", "purnea_0001.jpg"),
    ("cattle", "purnea", "https://www.dairyknowledge.in/sites/default/files/images/Purnea_Bull.jpg", "purnea_0002.jpg"),
    ("cattle", "kathani", "https://baif.org.in/wp-content/uploads/2023/10/Kathani-cattle.jpg", "kathani_0001.jpg"),
    ("cattle", "umarda", "https://cf-images.assettype.com/agrowon%2F2025-06-10%2F9lefthyy%2FNews-Story-Mahes-2025-06-10T093334.265.jpg", "umarda_0001.jpg"),
    ("cattle", "rohilkhandi", "https://www.researchgate.net/profile/Vallabhaneni-Srikanth/publication/404631474/figure/fig2/AS:11431282110513826@1778267439827/Rohilkhandi-cattle-with-its-owner_Q320.jpg", "rohilkhandi_0001.jpg")
]

raw_root = Path("dataset/raw")

for sp, folder, url, fname in downloads:
    dest_dir = raw_root / sp / folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / fname
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
            with open(dest_file, "wb") as f:
                f.write(content)
        # Verify with PIL
        with Image.open(dest_file) as im:
            w, h = im.size
        print(f"Downloaded {sp}/{folder}/{fname} -> {w}x{h} ({len(content)} bytes)")
    except Exception as e:
        print(f"Failed {url}: {e}")
