import urllib.request
import re

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; MCA Project AI Breed; mailto:mca.project@university.ac.in)'}

urls = [
    'https://nbagr.res.in/photo-gallery',
    'https://nbagr.res.in/gallery',
    'https://www.dahd.gov.in/cattle-and-dairy-development',
    'https://agricoop.nic.in'
]

for u in urls:
    try:
        req = urllib.request.Request(u, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"{u} -> {resp.status} (len {len(resp.read())})")
    except Exception as e:
        print(f"{u} -> {e}")
