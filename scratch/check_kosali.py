import urllib.request, ssl

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
redir = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHJIdLdmriLClWMkVhROH7HzQ7qMEtYtM1fd5vbuydaIVpMBQIFd2CsGESBJ-3OxLhHD_hFLF9CEmQ4jH3xeAHxrg7EofZ7DSMo0iDduh9psc52c-nC2wHDcVTa1VxFtjGZTwGGrr6q9ld1MyBUcQ6SkPDW7hFJm9h6A3NokFfhSD4KlOjtS70="

req = urllib.request.Request(redir, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        print("Final URL:", r.geturl())
except Exception as e:
    print("Error:", e)
