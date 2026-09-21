import urllib.request, ssl
from PIL import Image
import io

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}

fig_urls = [
    "https://arccarticles.s3.amazonaws.com/DOIArticleNew/Figure-images-B-5685-6089603e9087340b9bdcbbb1-1773045731873-0.jpg",
    "https://arccarticles.s3.amazonaws.com/DOIArticleNew/Figure-images-B-5685-6089603e9087340b9bdcbbb1-1773045731873-1.jpg",
    "https://arccarticles.s3.amazonaws.com/DOIArticleNew/Figure-images-B-5685-6089603e9087340b9bdcbbb1-1773045731873-2.jpg"
]

for idx, u in enumerate(fig_urls):
    req = urllib.request.Request(u, headers=headers)
    with urllib.request.urlopen(req, context=ctx) as r:
        data = r.read()
        im = Image.open(io.BytesIO(data))
        print(f"Fig {idx}: size={im.size}, mode={im.mode}, len={len(data)}")
        with open(f"scratch/melghati_fig_{idx}.jpg", "wb") as f:
            f.write(data)
