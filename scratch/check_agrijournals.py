import urllib.request, ssl

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
redir = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHsd8TZhcoTnBpg8l-db22v6sVE84k-yBZ3xQz7D7GSNxwC-MEXXDnV2dfuYg9Ft2MHjtJvUHwTkw6IULo4AqxlaaeBTLz27TioudSij66gdP1fyrvOKo-ZCVN-agq3Sh_auF1_CjvYXK_Ca30K4cS4AokcQpXjt9ONzL16BP5FPecbIrEg--vtZ5JWmrCTkh9yn6LhuKfOJC3FWARqjZpqrT0LtANb_sFHsd1wtuDeyQLJUKS39K57jI1aJPUuL51qGRXzbALRnFvs_F2UQ1JLTJhtNY9VlIJrlPYoTDkI6eP2ICYx"

req = urllib.request.Request(redir, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        print("Final URL:", r.geturl())
except Exception as e:
    print("Error:", e)
