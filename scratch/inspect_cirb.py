import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
url = 'https://cirb.res.in/buffalo-breeds/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, context=ctx) as r:
    h = r.read().decode('utf-8', errors='ignore')

imgs = re.findall(r'https://cirb\.res\.in/wp-content/uploads/[^\s"\'<>]+', h)
print(f'Total CIRB upload images: {len(imgs)}')
for im in sorted(set(imgs)):
    print(im)
