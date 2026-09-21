from pypdf import PdfReader
import re

reader = PdfReader("scratch/rohilkhandi_paper.pdf")
for i, p in enumerate(reader.pages):
    txt = p.extract_text()
    figs = re.findall(r'(Fig(?:ure)?\.?\s*\d+[^.\n]+(?:\.|\n))', txt, re.I)
    if figs:
        print(f"--- Page {i} Figures ---")
        for f in figs:
            print(f.strip())
