import urllib.request
import urllib.parse
import json
import time

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; AI Breed Recognition)'}

def get_wiki_page_image(title):
    url = f"https://en.wikipedia.org/w/api.php?action=query&titles={urllib.parse.quote(title)}&prop=pageimages|extracts&piprop=original|thumbnail&pithumbsize=600&exintro=1&explaintext=1&format=json"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            for pid, p in pages.items():
                if pid == "-1":
                    continue
                orig = p.get('original', {}).get('source', '')
                thumb = p.get('thumbnail', {}).get('source', '')
                return orig or thumb, p.get('title', ''), p.get('extract', '')[:150]
    except Exception as e:
        pass
    return None, None, None

sample_check = [
    "Bachaur cattle", "Khillari cattle", "Malnad Gidda cattle", "Kherigarh cattle",
    "Mewati cattle", "Marathwadi buffalo", "Ghumusari cattle", "Binjharpuri cattle",
    "Khariar cattle", "Kosali cattle", "Lakhimi cattle", "Konkan Kapila cattle",
    "Poda Thurpu cattle", "Dagri cattle", "Thutho cattle", "Shweta Kapila",
    "Himachali Pahari cattle", "Purnea cattle", "Kathani cattle", "Masilum cattle",
    "Rohilkhandi cattle", "Mahakaushali cattle", "Umarda cattle", "Melghati buffalo", "Gomanchali buffalo"
]

for title in sample_check:
    img_url, real_title, extract = get_wiki_page_image(title)
    print(f"'{title}': {img_url is not None} -> {img_url}")
