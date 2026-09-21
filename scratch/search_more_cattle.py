import urllib.request, urllib.parse, ssl, json, time

ctx = ssl._create_unverified_context()
USER_AGENT = "MCAResearchBot/1.0 (academic research; AI Breed Recognition; mailto:mca.project@university.ac.in)"
headers = {"User-Agent": USER_AGENT}

queries = {
    "pulikulam": ["Pulikulam", "Pulikulam bull", "Jallikattu bull"],
    "mewati": ["Mewati cow", "Mewati bull", "Kosi cattle"],
    "bachaur": ["Bachaur", "Sitamarhi cattle"],
    "belahi": ["Belahi cow", "Belahi bull"],
    "binjharpuri": ["Binjharpuri", "Jajpur cattle"],
    "ghumusari": ["Ghumusari", "Ghumsur cattle"],
    "kathani": ["Kathani cow", "Kathani bull"],
    "khariar": ["Khariar", "Khariar cow"],
    "koppal": ["Koppal cow", "Koppal bull"],
    "kosali": ["Kosali cow", "Kosali bull", "Chhattisgarh cow"],
    "umarda": ["Umarda cow", "Umarda bull"]
}

for folder, q_list in queries.items():
    print(f"\n=== Searching for {folder} ===")
    for q in q_list:
        api_url = (
            f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
            f"&gsrsearch={urllib.parse.quote(q)}&gsrnamespace=6&gsrlimit=5"
            f"&prop=imageinfo&iiprop=url|size|mime&format=json"
        )
        try:
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
                data = json.loads(r.read().decode('utf-8'))
                pages = data.get('query', {}).get('pages', {})
                for pid, p in pages.items():
                    title = p.get('title', '')
                    ii = p.get('imageinfo', [{}])[0]
                    url = ii.get('url', '')
                    mime = ii.get('mime', '')
                    if mime in ['image/jpeg', 'image/png'] and not any(k in title.lower() for k in ['map', 'flag', 'logo']):
                        print(f"[{q}] {title} : {url}")
            time.sleep(0.5)
        except Exception as e:
            print(f"[{q}] Error: {e}")
