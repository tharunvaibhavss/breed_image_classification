import urllib.request
import re

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; MCA Project AI Breed)'}
# Let's search NBAGR for monographs or breed pages
url = 'https://nbagr.res.in/?s=monograph+cattle'
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=15) as resp:
    html = resp.read().decode('utf-8', errors='ignore')

links = re.findall(r'<a\s+href=[\'"](.*?)[\'"].*?>(.*?)</a>', html, re.DOTALL)
print("NBAGR Search results:")
for href, title in links[:20]:
    title_clean = re.sub(r'<.*?>', '', title).strip()
    if title_clean and len(title_clean) > 3:
        print(f" - {title_clean}: {href}")
