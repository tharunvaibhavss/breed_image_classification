import urllib.request, ssl
from pypdf import PdfReader
from PIL import Image
import os, re

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
url = "https://dailypioneer.com/uploads/2026/epaper/august/bhopal-english-edition-2026-08-21.pdf"

req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
        pdf_data = resp.read()
        print(f"Downloaded PDF: {len(pdf_data)} bytes")
        with open("scratch/pioneer_bhopal.pdf", "wb") as f:
            f.write(pdf_data)

    reader = PdfReader("scratch/pioneer_bhopal.pdf")
    print(f"Total pages: {len(reader.pages)}")

    target_page = -1
    for idx, page in enumerate(reader.pages):
        txt = page.extract_text()
        if "mahakaushali" in txt.lower() or "cattle" in txt.lower() or "ndvsu" in txt.lower():
            print(f"Found mention on page {idx+1}")
            target_page = idx

    if target_page != -1:
        page = reader.pages[target_page]
        for img_idx, img in enumerate(page.images):
            name = f"scratch/pioneer_img_{img_idx}_{img.name}"
            with open(name, "wb") as fp:
                fp.write(img.data)
            im = Image.open(name)
            print(f"Image {img_idx}: {im.size}, {im.format}")
except Exception as e:
    print("Error:", e)
