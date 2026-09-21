import urllib.request
import re

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; AI Breed Recognition)'}

urls = [
    'https://www.dairyknowledge.in/article/indigenous-breeds',
    'https://www.dairyknowledge.in/article/cattle-breeds',
    'https://www.dairyknowledge.in/category/animal-breeds/cattle',
    'https://www.dairyknowledge.in/category/animal-breeds/buffalo'
]

for u in urls:
    try:
        req = urllib.request.Request(u, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            links = re.findall(r'<a[^>]+href=[\'"](/article/[^\'\"]+)[\'"][^>]*>(.*?)</a>', html)
            print(f"{u} (HTTP 200) -> found {len(links)} breed articles:")
            for l, name in links[:10]:
                print(f"   {name.strip()} -> https://www.dairyknowledge.in{l}")
    except Exception as e:
        print(f"{u} -> {e}")
