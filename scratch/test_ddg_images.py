import urllib.request
import urllib.parse
import json
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json'
}

def get_vqd(query):
    url = f"https://duckduckgo.com/?q={urllib.parse.quote(query)}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        m = re.search(r'vqd=([0-9-_]+)', html)
        if m:
            return m.group(1)
        m2 = re.search(r'vqd=[\'"]([^\'\"]+)', html)
        if m2:
            return m2.group(1)
    return None

def search_ddg_images(query):
    vqd = get_vqd(query)
    if not vqd:
        return []
    url = f"https://duckduckgo.com/i.js?l=us-en&o=json&q={urllib.parse.quote(query)}&vqd={vqd}&f=,,,&p=1"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        return data.get('results', [])

res = search_ddg_images("Ghumusari cattle breed")
print(f"Results: {len(res)}")
for r in res[:5]:
    print(" -", r.get('title'), "->", r.get('image'), "from", r.get('url'))
