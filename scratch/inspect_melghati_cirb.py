import urllib.request
import re

headers = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request("https://cirb.res.in/buffalo-breeds/", headers=headers)
with urllib.request.urlopen(req, timeout=15) as resp:
    html = resp.read().decode('utf-8', errors='ignore')

# find Melghati section
pos = html.find("Melghati")
if pos != -1:
    section = html[pos:pos+2000]
    print("Melghati section excerpt:")
    print(section[:1000])
    imgs = re.findall(r'<img[^>]+src=[\'"](.*?)[\'"]', section)
    print("Images in Melghati section:", imgs)
