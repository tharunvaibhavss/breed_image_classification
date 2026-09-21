import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}

url = 'https://icar.org.in/search/node?keys=Breed+Registration+Committee+Rohilkhandi'
req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        h = r.read().decode('utf-8', errors='ignore')
        # find search results
        items = re.findall(r'<h3 class="search-result__title"><a href="([^"]+)">([^<]+)</a>', h)
        if not items:
            items = re.findall(r'<a href="(/node/\d+)">([^<]+)</a>', h)
        for link, title in items:
            print(title.strip(), '-->', link)
except Exception as e:
    print('Error:', e)
