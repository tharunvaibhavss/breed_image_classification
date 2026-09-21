import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
url = "https://agrowon.esakal.com/agro-special/vidarbhas-umarda-cattle-gets-official-indigenous-breed-recognition-ds98"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, context=ctx) as r:
    h = r.read().decode('utf-8', errors='ignore')

og_imgs = re.findall(r'property="og:image"\s+content="([^"]+)"', h)
if not og_imgs:
    og_imgs = re.findall(r'content="([^"]+)"\s+property="og:image"', h)
print("og:image:", og_imgs)

# Also check for images in article body
body_imgs = re.findall(r'https://images\.assettype\.com/[^\s"\'<>]+', h)
print("assettype images:", len(body_imgs))
for im in set(body_imgs):
    print(im)
