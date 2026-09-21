import json
import urllib.request
import urllib.parse
import time
from pathlib import Path

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; MCA Project AI Breed; mailto:mca.project@university.ac.in)'}

with open('dataset/metadata/breed_details.json', 'r', encoding='utf-8') as f:
    breeds = json.load(f)

def search_commons(query, limit=20):
    url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={urllib.parse.quote(query)}&gsrnamespace=6&gsrlimit={limit}&prop=imageinfo&iiprop=url|size|extmetadata|mime&format=json"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            results = []
            for pid, p in pages.items():
                title = p.get('title', '')
                ii = p.get('imageinfo', [{}])[0]
                mime = ii.get('mime', '')
                if not mime.startswith('image/'):
                    continue
                if mime in ['image/svg+xml', 'image/x-xcf']:
                    continue
                results.append({
                    'title': title,
                    'url': ii.get('url', ''),
                    'mime': mime,
                    'license': ii.get('extmetadata', {}).get('LicenseShortName', {}).get('value', 'Unknown'),
                    'license_url': ii.get('extmetadata', {}).get('LicenseUrl', {}).get('value', ''),
                    'artist': ii.get('extmetadata', {}).get('Artist', {}).get('value', 'Unknown'),
                    'desc': ii.get('extmetadata', {}).get('ImageDescription', {}).get('value', ''),
                    'width': ii.get('width', 0),
                    'height': ii.get('height', 0)
                })
            return results
    except Exception as e:
        return []

summary = {}
for b in breeds:
    name = b['breed_name']
    sp = b['species']
    clean_name = name.split('(')[0].strip()
    q = f'"{clean_name}" {sp}'
    res = search_commons(q, limit=15)
    time.sleep(0.1)
    if not res:
        q2 = f'{clean_name} {sp}'
        res = search_commons(q2, limit=10)
        time.sleep(0.1)
    summary[name] = len(res)
    print(f"[{sp[:3].upper()}] {name}: {len(res)} candidate images")

with open('scratch/commons_search_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
print("Finished querying Wikimedia Commons.")
