import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
redir = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGKIyqet-s15RtebJjdf-7JqaAcF47uWV7Chczfrgv3nPVI6ZQkIgEaR3S0SXFWw7X5eQ8O_sQXixsi1ztfrQX5Bb12s5xjX54WDmfqzuuCgGz3Dl8oNDUmXRi99TnZswaNGKFCwTNcmu0-BCmWnCIg41AMYuuHKkwe"

req = urllib.request.Request(redir, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        print("Final URL:", r.geturl())
        h = r.read().decode('utf-8', errors='ignore')
        imgs = re.findall(r'src=["\']([^"\']+\.(?:jpg|jpeg|png))["\']', h, re.I)
        print("Images:", imgs)
except Exception as e:
    print("Error:", e)
