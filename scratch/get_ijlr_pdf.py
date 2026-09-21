import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
url = "https://ijlr.org/ojs_journal/index.php/ijlr/article/view/968/1393"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, context=ctx) as r:
    h = r.read().decode('utf-8', errors='ignore')

links = re.findall(r'href="([^"]+)"', h)
for l in links:
    if "download" in l or ".pdf" in l:
        print("Link:", l)
