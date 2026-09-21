import urllib.request

headers = {'User-Agent': 'Mozilla/5.0'}

urls = [
    "https://cirb.res.in/wp-content/uploads/2026/09/Melghati-female.jpg",
    "https://cirb.res.in/wp-content/uploads/2026/09/Melghati-male.jpg",
    "https://cirb.res.in/wp-content/uploads/2026/09/Melghati_female.jpg",
    "https://cirb.res.in/wp-content/uploads/2026/09/Melghati_Male.jpg",
    "https://cirb.res.in/wp-content/uploads/2026/09/Buffalo_Melghati_Female.jpg",
    "https://cirb.res.in/wp-content/uploads/2026/09/Buffalo_Melghati_Male.jpg",
    "https://cirb.res.in/wp-content/uploads/2026/09/Buffalo_Melghati_female.jpg",
    "https://cirb.res.in/wp-content/uploads/2026/09/Buffalo_Melghati_male.jpg",
    "https://cirb.res.in/wp-content/uploads/2026/09/Melghati.jpg"
]

for u in urls:
    try:
        req = urllib.request.Request(u, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"FOUND: {u} -> len {len(resp.read())}")
    except Exception:
        pass
