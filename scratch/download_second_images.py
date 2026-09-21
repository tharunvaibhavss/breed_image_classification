import urllib.request, ssl, os, time
from PIL import Image
import io

ctx = ssl._create_unverified_context()
headers = {
    'User-Agent': 'MCAResearchBot/1.0 (academic research; AI Breed Recognition; mailto:mca.project@university.ac.in)'
}

additional_images = [
    # Umblachery
    ("cattle", "umblachery", "umblachery_0002.jpg", "https://upload.wikimedia.org/wikipedia/commons/e/e9/Amblacheri_01.JPG"),
    # Sanchori
    ("cattle", "sanchori", "sanchori_0002.jpg", "https://upload.wikimedia.org/wikipedia/commons/2/24/Sanchori_cow_and_calf.jpg"),
    # Jaffarabadi
    ("buffalo", "jaffarabadi", "jaffarabadi_0002.jpg", "https://upload.wikimedia.org/wikipedia/commons/8/86/Brasilien_1992_25_%284607012976%29.jpg"),
]

for sp, folder, fname, url in additional_images:
    dest_dir = os.path.join("dataset", "raw", sp, folder)
    dest_path = os.path.join(dest_dir, fname)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"Already exists: {dest_path}")
        continue
    time.sleep(2)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            data = resp.read()
            im = Image.open(io.BytesIO(data))
            print(f"Downloaded {folder} ({im.size}, {im.format}): {len(data)} bytes")
            with open(dest_path, "wb") as f:
                f.write(data)
    except Exception as e:
        print(f"Failed {folder}: {e}")
