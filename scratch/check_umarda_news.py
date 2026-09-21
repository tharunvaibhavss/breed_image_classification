import urllib.request, ssl, re

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for name, redir in [
    ("esakal_1", "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEbAzTl0BWGDIkddG6nX5Te_GtFB2uPv_Lw_o3DcW53Fo2uPwI_H1WhdTFRSwkdQZ9DeomzyeO4zXlsyeh511nwmtrOhWzWmBD_5rSpP58rv33n8O8i2yjKz06Z2fAdiAsciKJomjv4qvYcJHk-1b4X6_0Lj_EERGAs0ghv6DRBgEtDXiHRRPqtrwMq6zGpmmmYGlJMYnMeR_RU0MgrGJMQgHf4"),
    ("esakal_2", "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEGg8_D_D-0IlIEaFhZDKxKwsibAZeTQqBYStXXqjLpLccLeHf_Nhp_SlJwfsam4NDD0rPPgVAMqoC6DjSG4_LMLNNJkTm9m4ikA4BLcl9wRuw1Wyo91fDx9bZjOjrudb5R_C2XpLwFmjTZQeEhNScZCf32HShQf5gvfDHvsJjsEfLqxoJQ23dZCdpSlcjD_9R5On2DdSVcbBJJUNVaLZvrEL34C0AcQbzjTbvyHauycg==")
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
