import urllib.request
import urllib.parse
import json
import re
import time

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; AI Breed Recognition)'}

missing_queries = {
    "Khillar": ["Khillari", "Killari", "Khillar bull", "Khillar cow", "Khillar cattle"],
    "Malnad Gidda": ["Malnad Gidda", "Malnad Gidda cow", "Gidda cattle", "Malnad Gidda cattle"],
    "Bachaur": ["Bachaur", "Bachhaur", "Bachaur cattle", "Bachaur bull"],
    "Kherigarh": ["Kherigarh", "Kheri cattle", "Kherigarh cattle"],
    "Mewati": ["Mewati cattle", "Mewati cow", "Kosi cattle"],
    "Marathwadi": ["Marathwadi buffalo", "Marathwadi", "Marathwada buffalo"],
    "Ghumusari": ["Ghumusari", "Ghumusari cattle", "Ghumusar cattle"],
    "Binjharpuri": ["Binjharpuri", "Binjharpuri cattle"],
    "Khariar": ["Khariar", "Khariar cattle"],
    "Kosali": ["Kosali cattle", "Kosali cow"],
    "Lakhimi": ["Lakhimi cattle", "Lakhimi cow"],
    "Konkan Kapila": ["Konkan Kapila", "Kapila cattle", "Konkan cattle"],
    "Poda Thurpu": ["Poda Thurpu", "Podathurpu", "Poda Thurpu cattle"],
    "Dagri": ["Dagri cattle", "Dahod cattle", "Dagri cow"],
    "Thutho": ["Thutho cattle", "Thutho Nagaland", "Thutho cow"],
    "Shweta Kapila": ["Shweta Kapila", "Shweta Kapila cow", "Goa Kapila"],
    "Himachali Pahari": ["Himachali Pahari", "Pahari cattle", "Himachali cattle", "Gaddi cattle"],
    "Purnea": ["Purnea cattle", "Purnea cow"],
    "Kathani": ["Kathani cattle", "Kathani cow"],
    "Masilum": ["Masilum cattle", "Meghalaya cattle"],
    "Rohilkhandi": ["Rohilkhandi cattle", "Rohilkhandi cow"],
    "Mahakaushali": ["Mahakaushali cattle", "Mahakaushal cattle"],
    "Umarda": ["Umarda cattle", "Umarda cow"],
    "Melghati": ["Melghati buffalo", "Gaolao buffalo", "Satpura buffalo"],
    "Gomanchali": ["Gomanchali buffalo", "Goan buffalo", "Gomantak buffalo"]
}

def search_commons(q):
    url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={urllib.parse.quote(q)}&gsrnamespace=6&gsrlimit=10&prop=imageinfo&iiprop=url|size|extmetadata|mime&format=json"
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
                if mime.startswith('image/') and mime not in ['image/svg+xml']:
                    results.append({'title': title, 'url': ii.get('url', ''), 'mime': mime, 'license': ii.get('extmetadata', {}).get('LicenseShortName', {}).get('value', 'Unknown')})
            return results
    except Exception as e:
        return []

found = {}
for breed, q_list in missing_queries.items():
    print(f"Searching for {breed}...")
    for q in q_list:
        res = search_commons(q)
        time.sleep(0.1)
        # filter out maps, diagrams
        clean_res = [r for r in res if not any(k in r['title'].lower() for k in ['map', 'flag', 'diagram', 'infographic', 'digest', 'pdf', 'document'])]
        if clean_res:
            print(f"  Found for query '{q}': {len(clean_res)} images")
            for r in clean_res[:3]:
                print(f"    - {r['title']} ({r['license']})")
            found.setdefault(breed, []).extend(clean_res)
            break

print(f"\nTotal breeds resolved: {len(found)} / {len(missing_queries)}")
