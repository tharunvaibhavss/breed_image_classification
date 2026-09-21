import urllib.request, ssl

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
redir = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHR0b7KCsAwkIHf2lrk0MhvtVAWB7sof-MrBI4GwPXRKJWwZzE0eOK1Pc6kLnXPvS_6Eh50lourKd_kv_l9xyvnE-W7VtQW5YpMU5eutiMpckFIWedZXTI_W7mrlxIN-ch1hjdyn6APURoNoX4="

req = urllib.request.Request(redir, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        print("Final URL:", r.geturl())
except Exception as e:
    print("Error:", e)
