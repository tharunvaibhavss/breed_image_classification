import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
redir = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG2O8ruk8wWfJ_0Zu6rgRcBG3gKaDdb5z-L8Z8YqzGE8dJ01lP2S9nxccCIsMxmoZ1b2WS5viqJqsVgGpopkfxDXj819jeM2RhgV3TgwS8jCECmU7Rqb1LQ_t5LDnqa4gJl_KWAtrsfKWCs-TOHeyi4NSITxnpOsRUOiWPUMQtVmTkl8aAfjYIH5qiwq1FSXAAbzfMoIDdDCjuWzlHr3jSqLQ=="

req = urllib.request.Request(redir, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        print("Final URL:", r.geturl())
        h = r.read().decode('utf-8', errors='ignore')
        imgs = re.findall(r'src=["\']([^"\']+\.(?:jpg|jpeg|png))["\']', h, re.I)
        print("Images:", imgs)
except Exception as e:
    print("Error:", e)
