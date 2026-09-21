import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
redir = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG0aOgwcBh3DA08FFHu9xEtVfTANl2P2JGvqC5eq5j3C1l44JIRlFmuLPQpY5IEsq9yk0Hc4I4NDE51809OH0zEJzXRI1WHrk9IRhEkBINi4mKmFnkZjrhXOyQD5uQ0_AjWgbUOXuJPUtg34V8qI4beq3Y="

req = urllib.request.Request(redir, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        print("Final URL:", r.geturl())
except Exception as e:
    print("Error:", e)
