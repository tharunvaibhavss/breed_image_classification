import urllib.request
import re

headers = {'User-Agent': 'Mozilla/5.0'}

# Test search icar.org.in
for term in ["Melghati", "Rohilkhandi", "Masilum"]:
    url = f"https://icar.org.in/search/node?keys={term}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            links = re.findall(r'<a[^>]+href=[\'"]([^\'\"]+)[\'"][^>]*>(.*?)</a>', html)
            print(f"\nTerm {term}:")
            for href, txt in links[:5]:
                if term.lower() in txt.lower() or 'breed' in txt.lower():
                    print(f"  - {txt.strip()} -> {href}")
    except Exception as e:
        print(f"Error for {term}: {e}")
