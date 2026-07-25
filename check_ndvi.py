import json
with open(r"C:\Users\madha\OneDrive\Desktop\yeild estimator\output\ndvi_data.json") as f:
    d = json.load(f)
for name in ["Bagalkot", "Ludhiana", "Jhunjhunu", "Idukki", "Pune", "Guntur"]:
    if name in d:
        x = d[name]
        raw = x.get("ndvi_raw", "?")
        fac = x.get("ndvi_factor", "?")
        src = x.get("source", "?")
        print(f"  {name}: raw={raw} factor={fac} source={src}")
