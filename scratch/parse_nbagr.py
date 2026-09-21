import urllib.request
import re
import json

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; MCA Project AI Breed)'}

def fetch_page(url):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode('utf-8', errors='ignore')

cattle_html = fetch_page('https://nbagr.res.in/cattle-breed')
buffalo_html = fetch_page('https://nbagr.res.in/node/114')

def parse_table(html):
    rows = re.findall(r'<tr.*?>(.*?)</tr>', html, re.DOTALL)
    data = []
    for r in rows:
        cols = re.findall(r'<t[dh].*?>(.*?)</t[dh]>', r, re.DOTALL)
        if not cols:
            continue
        cleaned = [re.sub(r'<.*?>', '', c).strip() for c in cols]
        links = re.findall(r'href=[\'"](.*?)[\'"]', r)
        data.append({'cols': cleaned, 'links': links})
    return data

cattle_rows = parse_table(cattle_html)
buffalo_rows = parse_table(buffalo_html)

print(f"Cattle parsed rows: {len(cattle_rows)}")
for i, r in enumerate(cattle_rows):
    if len(r['cols']) >= 3 and any('INDIA_CATTLE' in c for c in r['cols']):
        print(f"Cattle item: {r['cols']} | Links: {r['links']}")

print(f"\nBuffalo parsed rows: {len(buffalo_rows)}")
for i, r in enumerate(buffalo_rows):
    if len(r['cols']) >= 3 and any('INDIA_BUFFALO' in c for c in r['cols']):
        print(f"Buffalo item: {r['cols']} | Links: {r['links']}")
