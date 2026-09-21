import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
url = "https://epubs.icar.org.in/index.php/IJAnS/article/view/10262"

try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        h = r.read().decode('utf-8', errors='ignore')
        print("Page len:", len(h))
        links = re.findall(r'href="([^"]+)"', h)
        for l in links:
            if "download" in l or ".pdf" in l:
                print("PDF link:", l)
except Exception as e:
    print("Error:", e)
