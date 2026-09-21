import urllib.request, ssl
from pypdf import PdfReader
from PIL import Image
import io, os

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
pdf_url = "https://www.agriculturaljournals.com/archives/2026/vol8issue5/PartA/8-4-213-671.pdf"

req = urllib.request.Request(pdf_url, headers=headers)
with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
    pdf_bytes = resp.read()
    print(f"Downloaded PDF: {len(pdf_bytes)} bytes")
    with open("scratch/rohilkhandi_paper.pdf", "wb") as f:
        f.write(pdf_bytes)

reader = PdfReader("scratch/rohilkhandi_paper.pdf")
print(f"Total pages: {len(reader.pages)}")

os.makedirs("scratch/extracted_rohilkhandi", exist_ok=True)
count = 0
for page_num, page in enumerate(reader.pages):
    for img_idx, img_file in enumerate(page.images):
        count += 1
        name = f"scratch/extracted_rohilkhandi/page_{page_num}_{img_idx}_{img_file.name}"
        with open(name, "wb") as fp:
            fp.write(img_file.data)
        im = Image.open(name)
        print(f"Extracted: {name}, size: {im.size}, mode: {im.mode}")

print(f"Total images extracted: {count}")
