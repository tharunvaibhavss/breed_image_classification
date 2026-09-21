import urllib.request
import re

headers = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request("https://cirb.res.in/buffalo-breeds/", headers=headers)
with urllib.request.urlopen(req, timeout=15) as resp:
    html = resp.read().decode('utf-8', errors='ignore')

# Find all panels
panels = re.findall(r'<div class="vc_tta-panel[^"]*"(.*?)</div>\s*</div>\s*</div>\s*</div>\s*</div>', html, re.DOTALL)
print(f"Panels found: {len(panels)}")

# Or find all occurrences of "Melghati"
for m in re.finditer(r'Melghati', html, re.IGNORECASE):
    idx = m.start()
    snippet = html[idx-100:idx+400]
    print("\nOccurrence around Melghati:")
    print(snippet)
