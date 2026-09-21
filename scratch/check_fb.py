import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
url = "https://www.facebook.com/100015594514123/posts/ndvsu-jabalpur-mp-mahakaushali-cattle-registered-as-indias-57th-indigenous-cattl/2441579536371840/"
req = urllib.request.Request(url, headers={'User-Agent': 'facebookexternalhit/1.1'})
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        h = r.read().decode('utf-8', errors='ignore')
        og = re.findall(r'property="og:image"\s+content="([^"]+)"', h)
        if not og:
            og = re.findall(r'content="([^"]+)"\s+property="og:image"', h)
        print("og:image:", og)
except Exception as e:
    print("Error:", e)
