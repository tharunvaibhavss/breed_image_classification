import urllib.request
import ssl
import re

headers = {'User-Agent': 'Mozilla/5.0'}
ctx = ssl._create_unverified_context()

pages = [
    'https://icar.org.in/en/icar-nbagr-registers-16-new-livestock-and-poultry-breeds-strengthening-indias-animal-genetic',
    'https://icar.org.in/en/ten-new-breeds-indigenous-farm-animals-registered-icar-nbagr'
]

for p in pages:
    print(f"\n--- Checking {p} ---")
    req = urllib.request.Request(p, headers=headers)
    with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        imgs = re.findall(r'<img[^>]+src=[\'"](.*?)[\'"]', html)
        print(f"Images found: {len(imgs)}")
        for img in imgs:
            if not 'logo' in img and not 'icon' in img:
                print("  -", img)
