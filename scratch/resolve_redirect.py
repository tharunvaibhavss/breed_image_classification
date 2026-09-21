import urllib.request, ssl

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
redirect_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGyNIXTAGcO4J-w7KTTnkiIZkaLwcRXJkiX_aa0E1ypwvghUTfW9AN79OSRbQ5ppB4b04n_sDnOqeD_VyrutTzbtOPaJFCqrk8TcQHGaaqJlyZZob0Qv7JF3pJ8yVOgr5EpR4gT9cM3TKyc1hHQMcAr3pIgzwE-lryiQEElkb6NRfA4lhaHyXb4dM0rws0he6T-YgqHqX6hvlZ0_43NjcT3B_rda6RXTPbz"

req = urllib.request.Request(redirect_url, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
        print("Final URL:", resp.geturl())
        h = resp.read().decode('utf-8', errors='ignore')
        print(f"Content length: {len(h)}")
        import re
        imgs = re.findall(r'src=["\']([^"\']+\.(?:jpg|jpeg|png))["\']', h, re.I)
        print("Images:", imgs)
except Exception as e:
    print("Error:", e)
