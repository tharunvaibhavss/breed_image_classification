import urllib.request, ssl
from pypdf import PdfReader
from PIL import Image
import io, os

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
url = "https://veterinaryworld.org/Vol.18/January-2025/11.pdf"

req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
    data = resp.read()
    print(f"Downloaded Veterinary World PDF: {len(data)} bytes")
    with open("scratch/bachaur_paper.pdf", "wb") as f:
        f.write(data)

reader = PdfReader("scratch/bachaur_paper.pdf")
print(f"Total pages: {len(reader.pages)}")

os.makedirs("scratch/extracted_bachaur", exist_ok=True)
count = 0
for p_idx, page in enumerate(reader.pages):
    for i_idx, img in enumerate(page.images):
        name = f"scratch/extracted_bachaur/p_{p_idx}_{i_idx}_{img.name}"
        with open(name, "wb") as fp:
            fp.write(img.data)
        im = Image.open(name)
        print(f"Extracted {name}: {im.size}, {im.format}")
        count += 1
print(f"Total extracted: {count}")
