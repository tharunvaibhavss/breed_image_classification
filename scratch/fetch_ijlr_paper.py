import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
url = "https://ijlr.org/ojs_journal/index.php/ijlr/article/view/968"

req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
        h = r.read().decode('utf-8', errors='ignore')
        print("Page length:", len(h))
        pdfs = re.findall(r'href="([^"]+\.pdf[^"]*)"', h)
        print("PDFs:", pdfs)
        if not pdfs:
            pdfs = re.findall(r'href="(https?://ijlr\.org/[^"]+view/[^"]+)"', h)
            print("View links:", pdfs)
except Exception as e:
    print("Error:", e)
