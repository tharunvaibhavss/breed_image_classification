import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

url = 'https://cirb.res.in/buffalo-breeds/'
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
    html = resp.read().decode('utf-8', errors='ignore')

# Find panel with Melghati
m = re.search(r'id=["\']([^"\']+)["\'][^>]*>\s*<span[^>]*>Melghati Buffalo', html, re.I)
if m:
    tab_id = m.group(1)
    print('Found tab id:', tab_id)

pos = html.find('Melghati Buffalo</h2>')
if pos != -1:
    chunk = html[pos:pos+10000]
    imgs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', chunk)
    print('Imgs after Melghati Buffalo heading:', imgs)
else:
    print('Heading not found')
