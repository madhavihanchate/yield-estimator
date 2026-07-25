import json
with open(r"C:\Users\madha\OneDrive\Desktop\yeild estimator\output\ndvi_data.json") as f:
    d = json.load(f)
factors = [v["ndvi_factor"] for v in d.values() if v.get("ndvi_factor")]
factors.sort()
print(f"Districts: {len(factors)}")
print(f"Min: {factors[0]}")
print(f"P5: {factors[int(len(factors)*0.05)]}")
print(f"P25: {factors[int(len(factors)*0.25)]}")
print(f"Median: {factors[int(len(factors)*0.5)]}")
print(f"P75: {factors[int(len(factors)*0.75)]}")
print(f"P95: {factors[int(len(factors)*0.95)]}")
print(f"Max: {factors[-1]}")
below_3 = sum(1 for x in factors if x < 0.3)
below_5 = sum(1 for x in factors if x < 0.5)
print(f"Below 0.3: {below_3} districts ({below_3*100//len(factors)}%)")
print(f"Below 0.5: {below_5} districts ({below_5*100//len(factors)}%)")
# Show some extremes
low = sorted(d.items(), key=lambda x: x[1].get("ndvi_factor",1))[:5]
high = sorted(d.items(), key=lambda x: x[1].get("ndvi_factor",0), reverse=True)[:5]
print("\nLowest 5:")
for name, v in low:
    print(f"  {name}: {v['ndvi_factor']} (raw={v.get('ndvi_raw')})")
print("\nHighest 5:")
for name, v in high:
    print(f"  {name}: {v['ndvi_factor']} (raw={v.get('ndvi_raw')})")
