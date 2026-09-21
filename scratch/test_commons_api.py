import urllib.request
import urllib.parse
import json

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; MCA Project AI Breed; mailto:research@university.ac.in)'}

def search_commons(query, limit=10):
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
                img_url = ii.get('url', '')
                mime = ii.get('mime', '')
                extmeta = ii.get('extmetadata', {})
                license_name = extmeta.get('LicenseShortName', {}).get('value', 'Unknown')
                artist = extmeta.get('Artist', {}).get('value', 'Unknown')
                desc = extmeta.get('ImageDescription', {}).get('value', '')
                results.append({
                    'title': title,
                    'url': img_url,
                    'mime': mime,
                    'license': license_name,
                    'artist': artist,
                    'width': ii.get('width', 0),
                    'height': ii.get('height', 0)
                })
            return results
    except Exception as e:
        print(f"Error querying {query}: {e}")
        return []

for q in ["Amritmahal cattle", "Gir cattle", "Ongole cattle", "Murrah buffalo", "Toda buffalo"]:
    res = search_commons(q, limit=5)
    print(f"\nQuery: '{q}' -> found {len(res)} results")
    for r in res[:2]:
        print(f"  Title: {r['title']}")
        print(f"  URL: {r['url']}")
        print(f"  License: {r['license']} | Dimensions: {r['width']}x{r['height']}")
