import urllib.request
import urllib.parse
import json

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; MCA Project AI Breed; mailto:mca.project@university.ac.in)'}

def search_commons(query, limit=20):
    url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={urllib.parse.quote(query)}&gsrnamespace=6&gsrlimit={limit}&prop=imageinfo&iiprop=url|size|extmetadata|mime&format=json"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            results = []
            for pid, p in pages.items():
                title = p.get('title', '')
                ii = p.get('imageinfo', [{}])[0]
                results.append({
                    'title': title,
                    'url': ii.get('url', ''),
                    'mime': ii.get('mime', ''),
                    'license': ii.get('extmetadata', {}).get('LicenseShortName', {}).get('value', 'Unknown'),
                    'width': ii.get('width', 0),
                    'height': ii.get('height', 0)
                })
            return results
    except Exception as e:
        print(f"Error {query}: {e}")
        return []

sample_breeds = [
    ("Amritmahal", "cattle"),
    ("Bargur", "cattle"),
    ("Deoni", "cattle"),
    ("Hallikar", "cattle"),
    ("Kangayam", "cattle"),
    ("Kankrej", "cattle"),
    ("Punganur", "cattle"),
    ("Vechur", "cattle"),
    ("Bhadawari", "buffalo"),
    ("Jaffarabadi", "buffalo"),
    ("Pandharpuri", "buffalo"),
    ("Chilika", "buffalo"),
    ("Banni", "buffalo")
]

for name, sp in sample_breeds:
    q = f"{name} {sp}"
    res = search_commons(q, limit=15)
    print(f"{name} ({sp}): {len(res)} results")
    for r in res[:2]:
        print(f"   {r['title']} [{r['width']}x{r['height']}] ({r['license']})")
