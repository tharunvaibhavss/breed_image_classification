import urllib.request
import re

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; AI Breed Recognition)'}

url = "https://saveindiancows.org/indigenous-cattle-breeds-of-india/"
try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        imgs = re.findall(r'<img[^>]+src=[\'"](.*?)[\'"]', html)
        print(f"saveindiancows page has {len(imgs)} images:")
        for img in imgs:
            if 'uploads' in img:
                print(" -", img)
except Exception as e:
    print(f"Error: {e}")
