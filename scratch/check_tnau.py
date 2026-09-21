import urllib.request
import re

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; AI Breed Recognition)'}

url = "http://agritech.tnau.ac.in/animal_husbandry/ani_cat_cattle_breed.html"
try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        imgs = re.findall(r'<img[^>]+src=[\'"](.*?)[\'"]', html)
        print(f"TNAU cattle breeds page has {len(imgs)} images:")
        for img in imgs:
            if not 'header' in img and not 'banner' in img:
                print(" -", img)
except Exception as e:
    print(f"Error fetching TNAU: {e}")
