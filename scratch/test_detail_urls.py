import urllib.request
import re

headers = {'User-Agent': 'MCAResearchBot/1.0 (academic research; MCA Project AI Breed)'}

test_urls = [
    'https://nbagr.res.in/registered-cattle-breed',
    'https://nbagr.res.in/gir-cattle',
    'https://nbagr.res.in/?s=Gir',
    'https://cirb.res.in/murrah/',
    'https://cirb.res.in/buffalo-breeds/'
]

for url in test_urls:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"{url} -> {resp.status} (len {len(resp.read())})")
    except Exception as e:
        print(f"{url} -> FAILED ({e})")
