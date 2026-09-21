import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
redir = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGin7TMyCSnGdnAVnp6wKpGoDMkrG6JUkbMsRnu-mMVykAF8rspLoHz4Q3NS78nW-BOv3iWaZKV9ZfnQKQ_GpoDXXz_-Fut7xTuLsYqOtvjKs4LKGrxz2zN0Uyvrr_x4u8N3vtmGJ9Id5brqNj0bbts71mgDZ48NS1rHlbvNMhn3tnjegJZUESSVA=="

req = urllib.request.Request(redir, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        print("Final URL:", r.geturl())
        h = r.read().decode('utf-8', errors='ignore')
        imgs = re.findall(r'src=["\']([^"\']+\.(?:jpg|jpeg|png))["\']', h, re.I)
        print("Images:", imgs)
except Exception as e:
    print("Error:", e)
