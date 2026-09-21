import urllib.request
import json
import time

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

missing = [
    "Kherigarh", "Mewati", "Khariar", "Malnad Gidda", "Lakhimi",
    "Konkan Kapila", "Poda Thurpu", "Dagri", "Thutho", "Shweta Kapila",
    "Himachali Pahari", "Purnea", "Kathani", "Masilum", "Rohilkhandi",
    "Mahakaushali", "Umarda", "Melghati", "Gomanchali"
]

found = {}

# Test various months and patterns on saveindiancows
years_months = [
    "2018/01", "2018/02", "2018/03", "2018/04", "2018/05",
    "2017/12", "2017/11", "2019/01", "2019/02", "2019/03"
]

for b in missing:
    b_clean = b.lower().replace(' ', '_')
    b_upper = b.upper().replace(' ', '_')
    names = [
        f"Cow_{b_upper}.png",
        f"Cow_{b_upper}.jpg",
        f"Bull_{b_upper}.png",
        f"Bull_{b_upper}.jpg",
        f"{b_clean}.png",
        f"{b_clean}.jpg",
        f"Cow_{b}.png",
        f"Cow_{b}.jpg"
    ]
    for ym in years_months:
        for n in names:
            url = f"https://saveindiancows.org/wp-content/uploads/{ym}/{n}"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=3) as resp:
                    if resp.status == 200:
                        print(f"FOUND SIC: {b} -> {url}")
                        found.setdefault(b, []).append(url)
            except Exception:
                pass
            time.sleep(0.02)
        if b in found:
            break

print(f"Resolved on saveindiancows: {len(found)}")
