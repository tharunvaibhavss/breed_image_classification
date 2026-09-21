import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}

# Check NDDB cattle breeds page
url = "https://www.dairyknowledge.in/article/cattle-breeds"
try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        h = r.read().decode('utf-8', errors='ignore')
        links = re.findall(r'href="(/article/[^"]+)"[^>]*>([^<]+)</a>', h)
        print("NDDB cattle links:", len(links))
        for l, t in links[:15]:
            print(f"{t.strip()} -> {l}")
except Exception as e:
    print("NDDB cattle error:", e)

# Check NDDB buffalo breeds page
url2 = "https://www.dairyknowledge.in/article/buffalo-breeds"
try:
    req = urllib.request.Request(url2, headers=headers)
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        h = r.read().decode('utf-8', errors='ignore')
        links = re.findall(r'href="(/article/[^"]+)"[^>]*>([^<]+)</a>', h)
        print("NDDB buffalo links:", len(links))
        for l, t in links:
            print(f"{t.strip()} -> {l}")
except Exception as e:
    print("NDDB buffalo error:", e)
