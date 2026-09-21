import urllib.request
import re

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; MCA Project AI Breed; mailto:mca.project@university.ac.in)'}
req = urllib.request.Request('https://cirb.res.in/buffalo-breeds/', headers=headers)
with urllib.request.urlopen(req, timeout=15) as resp:
    html = resp.read().decode('utf-8', errors='ignore')
    imgs = re.findall(r'<img[^>]+src=[\'"](.*?)[\'"]', html, re.IGNORECASE)
    for i, img in enumerate(imgs):
        if 'uploads' in img and not 'logo' in img and not 'flag' in img:
            print(f"{i}: {img}")
