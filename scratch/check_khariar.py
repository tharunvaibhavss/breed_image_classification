import urllib.request, ssl

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
redir = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHyNZ18Wlbdd4D5Gc8ti3-vSBoiWEv1-O2I-Mlf2I7YIUo2KsBKzEUsvx0J4zLz4Me3RQbjal72OPipxwpoThtM4X-T5lmHNUCWGWpRwwK3mAeBXXkAwjIcav2JlNoJryV94J_gMZoFDYVez1EPf3XqahnQ4C4OEgpfG4DCMIpD"

req = urllib.request.Request(redir, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        print("Final URL:", r.geturl())
except Exception as e:
    print("Error:", e)
