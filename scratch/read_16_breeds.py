import urllib.request, ssl, re, sys

sys.stdout.reconfigure(encoding='utf-8')
ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0'}
url = "https://icar.org.in/en/icar-nbagr-registers-16-new-livestock-and-poultry-breeds-strengthening-indias-animal-genetic"

req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, context=ctx) as resp:
    h = resp.read().decode('utf-8', errors='ignore')

# strip HTML tags to read the text
text = re.sub(r'<[^>]+>', ' ', h)
text = ' '.join(text.split())
print("\n--- Mentions of Rohilkhandi or Melghati ---")
for breed in ['Rohilkhandi', 'Melghati']:
    pos = text.find(breed)
    if pos != -1:
        print(text[pos-100:pos+300])
        print("="*40)

# Check images and links in page
imgs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', h)
print("Images found:")
for img in imgs:
    print(img)
