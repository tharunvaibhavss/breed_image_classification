import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
url = "https://www.ndvsu.org/index.php"

try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        h = r.read().decode('utf-8', errors='ignore')
        imgs = re.findall(r'src=["\']([^"\']+)["\']', h)
        for img in imgs:
            if any(k in img.lower() for k in ['cattle', 'cow', 'breed', 'mahakaushali', 'achievement']):
                print(img)
except Exception as e:
    print("Error:", e)
