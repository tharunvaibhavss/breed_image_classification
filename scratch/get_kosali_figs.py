import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
url = "https://arccjournals.com/journal/asian-journal-of-dairy-and-food-research/DR-1973"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, context=ctx) as r:
    h = r.read().decode('utf-8', errors='ignore')

imgs = re.findall(r'https://arccarticles\.s3\.amazonaws\.com/DOIArticleNew/[^\s"\'<>]+', h)
print("Kosali images found:", len(imgs))
for img in set(imgs):
    print(img)
