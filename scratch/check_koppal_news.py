import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for name, redir in [
    ("kannadaprabha", "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE0nblt4FGPVmn-fHmaVx6BQcHHkWKZ3fI2wEaZOtUnEA7BzfBF55faTmiXDABo1Ucle3ZnMVR8RIzlFazI_BQXeAzvlS7Q9owpWT6PQmZEkL2cqW64ud1WkszqeifjW3gkq-1gLe7-e81b_-oEEBKPyrZrge7pyaTEWAPD9I7crxA_E-DwE8E97dt4zkLfhE0ZSFdvyqHwGy6BsowxQHV7nCf7"),
    ("kannadanet", "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHJzCVq_vSQtO_C-GzIjGj0z7B9Npjb3clSpIb2wtej_ozu0phXNbtD9E_zJ1e_lvy4mTVBIYjd80PS1q2AvuxFf3SmJ19-YTKmTWphgXT37jQPvWfk")
]:
    try:
        req = urllib.request.Request(redir, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
            final_url = r.geturl()
            print(f"{name} Final URL:", final_url)
            h = r.read().decode('utf-8', errors='ignore')
            imgs = re.findall(r'src=["\']([^"\']+\.(?:jpg|jpeg|png))["\']', h, re.I)
            print(f"{name} Images:", imgs[:5])
    except Exception as e:
        print(f"{name} error: {e}")
