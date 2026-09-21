import urllib.request
import re

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; MCA Project AI Breed)'}
req = urllib.request.Request('https://cirb.res.in/buffalo-breeds/', headers=headers)
with urllib.request.urlopen(req, timeout=15) as resp:
    html = resp.read().decode('utf-8', errors='ignore')

headings = re.findall(r'<h[234].*?>(.*?)</h[234]>', html, re.DOTALL)
print("Headings found in CIRB buffalo breeds:")
for h in headings[:25]:
    clean = re.sub(r'<.*?>', '', h).strip()
    if clean:
        print(" -", clean)
