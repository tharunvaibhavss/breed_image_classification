import urllib.request, ssl

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
redir = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGwe5yeXhcfWt85qNz8_q3j69N10joUhihuXLAVLWQX0mAvf-woY3P-aI3j9vDkxO8Ef6UzbQeAoZ1oc4qxSZQmXkznLlTmPk4DSZ_Wz-Y5KZnkDkbEX2kP5pr1IvjcRR6OjczWo5g3qsyRvPkduQGJ5fuPS6ucAj2HJ80KVOE="

req = urllib.request.Request(redir, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        print("Final URL:", r.geturl())
except Exception as e:
    print("Error:", e)
