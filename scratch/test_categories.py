import urllib.request
import urllib.parse
import json

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; MCA Project AI Breed; mailto:mca.project@university.ac.in)'}

def get_category_members(cat_name):
    url = f"https://commons.wikimedia.org/w/api.php?action=query&list=categorymembers&cmtitle={urllib.parse.quote(cat_name)}&cmlimit=50&format=json"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get('query', {}).get('categorymembers', [])
    except Exception as e:
        print(f"Error querying {cat_name}: {e}")
        return []

cats = [
    "Category:Cattle breeds of India",
    "Category:Zebu breeds",
    "Category:Water buffalo breeds",
    "Category:Water buffaloes in India"
]

for cat in cats:
    members = get_category_members(cat)
    print(f"\n{cat}: {len(members)} members")
    for m in members[:15]:
        print(f"  - {m.get('title')} ({m.get('type')})")
