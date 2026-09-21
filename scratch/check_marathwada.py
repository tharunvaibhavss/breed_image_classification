import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
url = "https://cirb.res.in/marathwada/"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, context=ctx) as r:
    h = r.read().decode('utf-8', errors='ignore')

imgs = re.findall(r'src=["\']([^"\']+\.(?:jpg|jpeg|png))["\']', h, re.I)
for img in set(imgs):
    if "upload" in img.lower():
        print(img)
