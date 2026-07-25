import urllib.request, json, time
t = time.time()
r = urllib.request.urlopen('http://127.0.0.1:8000/api/predict/?district=Bagalkot&crop=Brinjal&soc=0.5&pH=7.2&N=180&P=12&K=130', timeout=60)
d = json.loads(r.read())
el = time.time() - t
x = d['data']
print(f"OK in {el:.1f}s")
print(f"Predicted: {x['predicted_yield']} kg/ha")
print(f"NDVI source: {x['ndvi_source']}")
print(f"NDVI factor: {x['ndvi_factor']}")
print(f"NDVI raw: {x['ndvi_raw']}")
print(f"GEE auth: {x['gee_authenticated']}")
