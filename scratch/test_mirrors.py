import urllib.request

headers = {'User-Agent': 'Mozilla/5.0'}

# Test if we can fetch Masilum, Melghati, Rohilkhandi
urls = [
    "https://imgv2-2-f.scribdassets.com/img/document/782041538/original/08b61ec22f/1?v=1", # ICAR-NBAGR Annual report with Masilum
    "https://cf-images.assettype.com/agrowon%2F2025-06-10%2F9lefthyy%2FNews-Story-Mahes-2025-06-10T093334.265.jpg",
    "https://esakal.com"
]

for u in urls:
    try:
        req = urllib.request.Request(u, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            print("OK:", u[:60], resp.status, len(resp.read()))
    except Exception as e:
        print("ERR:", u[:60], e)
