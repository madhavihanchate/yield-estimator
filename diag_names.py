import json

with open(r"C:\Users\madha\OneDrive\Desktop\yeild estimator\output\lookup.json") as f:
    lookup = json.load(f)

# Check wheat, rice, cardamom, sugarcane, chilli crops
targets = ["wheat", "rice", "cardamom", "sugarcane", "chilli", "chili"]
for t in targets:
    matches = [c for c in lookup["crops"] if t in c.lower()]
    print(f"'{t}': {matches[:8]}")

# Check what districts have data for some crops
with open(r"C:\Users\madha\OneDrive\Desktop\yeild estimator\output\potential_yields.json") as f:
    pot = json.load(f)

# Find Wheat entries
wheat_keys = [k for k in pot if "|Wheat" in k or "|wheat" in k.lower()]
print(f"\nWheat pairs: {len(wheat_keys)}")
for k in wheat_keys[:10]:
    print(f"  {k}: mean={pot[k]['mean']:.0f}")

# Rice
rice_keys = [k for k in pot if "|Rice" in k]
print(f"\nRice pairs: {len(rice_keys)}")
for k in rice_keys[:10]:
    print(f"  {k}: mean={pot[k]['mean']:.0f}")

# Chilli
chilli_keys = [k for k in pot if "chilli" in k.lower() or "chili" in k.lower()]
print(f"\nChilli pairs: {len(chilli_keys)}")
for k in chilli_keys[:10]:
    print(f"  {k}: mean={pot[k]['mean']:.0f}")
