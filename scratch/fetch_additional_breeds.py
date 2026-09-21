import urllib.request, urllib.parse, json, ssl, re, os
from PIL import Image
import io

ctx = ssl._create_unverified_context()
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
HEADERS = {"User-Agent": USER_AGENT}

breeds_to_check = [
    ("cattle", "bachaur", "Bachaur cattle"),
    ("cattle", "belahi", "Belahi cattle"),
    ("cattle", "binjharpuri", "Binjharpuri cattle"),
    ("cattle", "ghumusari", "Ghumusari cattle"),
    ("cattle", "kathani", "Kathani cattle"),
    ("cattle", "khariar", "Khariar cattle"),
    ("cattle", "kherigarh", "Kherigarh cattle"),
    ("cattle", "khillar", "Khillari cattle"),
    ("cattle", "koppal", "Koppal cattle"),
    ("cattle", "kosali", "Kosali cattle"),
    ("cattle", "mewati", "Mewati cattle"),
    ("cattle", "pulikulam", "Pulikulam cattle"),
    ("cattle", "sanchori", "Sanchori cattle"),
    ("cattle", "umarda", "Umarda cattle"),
    ("cattle", "umblachery", "Umblachery cattle"),
    ("buffalo", "jaffarabadi", "Jaffarabadi buffalo"),
    ("buffalo", "marathwadi", "Marathwadi buffalo")
]

for sp, folder, q in breeds_to_check:
    api_url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={urllib.parse.quote(q)}&gsrnamespace=6&gsrlimit=10"
        f"&prop=imageinfo&iiprop=url|size|mime&format=json"
    )
    try:
        req = urllib.request.Request(api_url, headers=HEADERS)
        with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
            data = json.loads(r.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            print(f"{folder} ({q}): found {len(pages)} candidates on Commons")
            for pid, p in pages.items():
                title = p.get('title', '')
                ii = p.get('imageinfo', [{}])[0]
                url = ii.get('url', '')
                mime = ii.get('mime', '')
                if mime in ['image/jpeg', 'image/png'] and not any(k in title.lower() for k in ['map', 'flag', 'logo', 'drawing']):
                    print(f"   -> {title} : {url}")
    except Exception as e:
        print(f"{folder} error: {e}")
