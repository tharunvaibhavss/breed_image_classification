import urllib.request
import urllib.parse
import json

headers = {'User-Agent': 'MCAResearchBot/1.0'}

queries = [
    "Mewati", "Khariar", "Masilum", "Mahakaushali", "Rohilkhandi", "Melghati", "Gomanchali"
]

for q in queries:
    url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={urllib.parse.quote(q)}&gsrnamespace=6&gsrlimit=5&prop=imageinfo&iiprop=url|mime&format=json"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            print(f"\nQuery '{q}': found {len(pages)} items")
            for pid, p in pages.items():
                title = p.get('title', '')
                ii = p.get('imageinfo', [{}])[0]
                print(f"  - {title} ({ii.get('mime')})")
    except Exception as e:
        print(f"Error for {q}: {e}")
