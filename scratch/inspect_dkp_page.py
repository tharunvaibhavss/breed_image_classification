import urllib.request
import re

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; AI Breed Recognition)'}
req = urllib.request.Request('https://www.dairyknowledge.in/article/indigenous-breeds', headers=headers)
with urllib.request.urlopen(req, timeout=10) as resp:
    html = resp.read().decode('utf-8', errors='ignore')
    # find images
    imgs = re.findall(r'<img[^>]+src=[\'"](.*?)[\'"]', html)
    print("Images on indigenous-breeds:")
    for img in imgs:
        print(" -", img)
    # find links
    links = re.findall(r'<a[^>]+href=[\'"](.*?)[\'"][^>]*>(.*?)</a>', html)
    print("\nRelevant links:")
    for href, txt in links:
        if any(w in txt.lower() for w in ['cattle', 'buffalo', 'cow', 'breed']):
            print(f" - {txt.strip()} -> {href}")
