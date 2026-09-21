import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
redir = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG92_1r57cbQrPfdzR1xI28g3qcZLMktLC_gt5M_M1xMxtQM4iYSkp3ALwTcosl6NY2RTunMls9ezgjhR8fE1CwVIieD63fjdUzY8hEubdfGA1LmyCgh5DpsN9ro4ehTlwpx1watA2MUF7ZEYhQXElM1aTjphjtV3X6kvFzFjiSc6IeNZ6VxHSKHNPbQNhVlw=="

req = urllib.request.Request(redir, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        print("Final URL:", r.geturl())
        h = r.read().decode('utf-8', errors='ignore')
        imgs = re.findall(r'src=["\']([^"\']+\.(?:jpg|jpeg|png))["\']', h, re.I)
        print("Images:", imgs)
except Exception as e:
    print("Error:", e)
