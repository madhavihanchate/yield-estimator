import json

with open(r"C:\Users\madha\OneDrive\Desktop\yeild estimator\output\hierarchy.json") as f:
    h = json.load(f)

# Full crop list for Ludhiana
ludhiana_crops = h.get("Punjab", {}).get("Ludhiana", [])
print(f"Ludhiana crops ({len(ludhiana_crops)}):")
for c in ludhiana_crops:
    print(f"  {c}")

# Check if wheat is there at all
wheat = [c for c in ludhiana_crops if "heat" in c.lower()]
print(f"\nWheat in Ludhiana hierarchy: {wheat}")

# Check potential_yields for Ludhiana
with open(r"C:\Users\madha\OneDrive\Desktop\yeild estimator\output\potential_yields.json") as f:
    pot = json.load(f)
lud_all = [k.split("|")[1] for k in pot if k.startswith("Ludhiana|")]
print(f"\nPotential yields Ludhiana crops ({len(lud_all)}):")
for c in sorted(set(lud_all)):
    print(f"  {c}")
