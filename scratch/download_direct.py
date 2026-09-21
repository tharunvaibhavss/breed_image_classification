import urllib.request
import urllib.parse
import json
from pathlib import Path

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; AI Breed Recognition)'}

# Targeted direct additions
additional_downloads = [
    ("cattle", "bachaur", "https://www.dairyknowledge.in/dkp/sites/default/files/styles/medium_large/public/bachaur_cow.jpg", "bachaur_0001.jpg"),
    ("cattle", "khillar", "https://www.dairyknowledge.in/dkp/sites/default/files/styles/medium_large/public/khillar_bull.jpg", "khillar_0001.jpg"),
    ("cattle", "ghumusari", "https://www.dairyknowledge.in/dkp/sites/default/files/styles/medium_large/public/ghumusari_cow.png", "ghumusari_0001.png"),
    ("cattle", "binjharpuri", "https://www.dairyknowledge.in/dkp/sites/default/files/styles/medium_large/public/binjharpuri_cow.png", "binjharpuri_0001.png"),
    ("cattle", "kosali", "https://www.dairyknowledge.in/dkp/sites/default/files/styles/medium_large/public/kosali.jpg", "kosali_0001.jpg"),
    ("buffalo", "marathwadi", "https://www.dairyknowledge.in/dkp/sites/default/files/styles/medium_large/public/marathwadi.jpg", "marathwadi_0001.jpg"),
]

raw_root = Path("dataset/raw")

for sp, folder, url, fname in additional_downloads:
    dest_dir = raw_root / sp / folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / fname
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
            with open(dest_file, "wb") as f:
                f.write(content)
        print(f"Downloaded {sp}/{folder}/{fname} ({len(content)} bytes)")
    except Exception as e:
        print(f"Failed {url}: {e}")

# Also fetch Malenadu Gidda and Kerigar from Wikimedia Commons
def get_commons_direct(title, sp, folder, fname):
    api_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={urllib.parse.quote(title)}&prop=imageinfo&iiprop=url&format=json"
    req = urllib.request.Request(api_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            for pid, p in pages.items():
                img_url = p.get('imageinfo', [{}])[0].get('url')
                if img_url:
                    dest_dir = raw_root / sp / folder
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = dest_dir / fname
                    req2 = urllib.request.Request(img_url, headers=headers)
                    with urllib.request.urlopen(req2, timeout=20) as resp2:
                        content = resp2.read()
                        with open(dest_file, "wb") as f:
                            f.write(content)
                    print(f"Downloaded Commons: {sp}/{folder}/{fname} ({len(content)} bytes)")
    except Exception as e:
        print(f"Error fetching {title}: {e}")

get_commons_direct("File:Malenadu Gidda.jpg", "cattle", "malnad_gidda", "malnad_gidda_0001.jpg")
get_commons_direct("File:Malenadu Gidda 01.JPG", "cattle", "malnad_gidda", "malnad_gidda_0002.jpg")
get_commons_direct("File:Malenadu Gidda 02.JPG", "cattle", "malnad_gidda", "malnad_gidda_0003.jpg")
get_commons_direct("File:Kerigar 01.JPG", "cattle", "kherigarh", "kherigarh_0001.jpg")
get_commons_direct("File:Kerigar 02.JPG", "cattle", "kherigarh", "kherigarh_0002.jpg")
