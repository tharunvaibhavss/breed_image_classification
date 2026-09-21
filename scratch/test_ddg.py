import urllib.request
import urllib.parse
import json
import re

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

def duckduckgo_search_images(query, max_results=5):
    # DDG image search
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            links = re.findall(r'<a class="result__url"[^>]+href=[\'"](.*?)[\'"]', html)
            return links[:max_results]
    except Exception as e:
        return [str(e)]

print("Test DDG:", duckduckgo_search_images("Ghumusari cattle site:icar.gov.in OR site:nbagr.res.in OR site:nic.in"))
