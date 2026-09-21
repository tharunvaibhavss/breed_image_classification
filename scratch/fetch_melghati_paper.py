import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
url = "https://arccjournals.com/journal/indian-journal-of-animal-research/B-5685"

req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
        h = r.read().decode('utf-8', errors='ignore')
        print("Page length:", len(h))
        imgs = re.findall(r'src=["\']([^"\']+\.(?:jpg|jpeg|png))["\']', h, re.I)
        print("Images:", imgs)
        pdfs = re.findall(r'href=["\']([^"\']+\.pdf)["\']', h, re.I)
        print("PDFs:", pdfs)
except Exception as e:
    print("Error:", e)
