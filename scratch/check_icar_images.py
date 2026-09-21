import urllib.request
import re

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; MCA Project AI Breed; mailto:mca.project@university.ac.in)'}

def get_images_from_url(url):
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            # find img src
            imgs = re.findall(r'<img[^>]+src=[\'"](.*?)[\'"]', html, re.IGNORECASE)
            return imgs
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

cirb_imgs = get_images_from_url('https://cirb.res.in/buffalo-breeds/')
print(f"CIRB Buffalo Breeds page has {len(cirb_imgs)} images:")
for img in cirb_imgs[:15]:
    print("  -", img)

nbagr_c_imgs = get_images_from_url('https://nbagr.res.in/cattle-breed')
print(f"\nNBAGR Cattle Breeds page has {len(nbagr_c_imgs)} images:")
for img in nbagr_c_imgs[:15]:
    print("  -", img)

nbagr_b_imgs = get_images_from_url('https://nbagr.res.in/node/114')
print(f"\nNBAGR Buffalo Breeds page has {len(nbagr_b_imgs)} images:")
for img in nbagr_b_imgs[:15]:
    print("  -", img)
