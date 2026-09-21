import urllib.request
import re

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; MCA Project AI Breed)'}
req = urllib.request.Request('https://nbagr.res.in/cattle-breed', headers=headers)
with urllib.request.urlopen(req, timeout=15) as resp:
    html = resp.read().decode('utf-8', errors='ignore')

rows = re.findall(r'<tr.*?>(.*?)</tr>', html, re.DOTALL)
cattle_records = []
for r in rows:
    cols = re.findall(r'<t[dh].*?>(.*?)</t[dh]>', r, re.DOTALL)
    cleaned = [re.sub(r'<.*?>', '', c).strip() for c in cols]
    if len(cleaned) >= 3 and any('INDIA_CATTLE' in c for c in cleaned):
        # find which col has accession, name, state
        accession = [c for c in cleaned if 'INDIA_CATTLE' in c][0]
        name = cleaned[1] if len(cleaned) > 1 else ''
        state = cleaned[2] if len(cleaned) > 2 else ''
        cattle_records.append((accession, name, state))

print(f"Total verified cattle records: {len(cattle_records)}")
for acc, name, state in cattle_records:
    print(f"{acc} | {name} | {state}")
