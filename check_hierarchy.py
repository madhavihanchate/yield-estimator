import json

with open(r"C:\Users\madha\OneDrive\Desktop\yeild estimator\output\hierarchy.json") as f:
    h = json.load(f)

# Check a few states/districts
for state in ["Punjab", "Maharashtra", "Andhra Pradesh", "Rajasthan"]:
    if state not in h:
        print(f"STATE NOT FOUND: {state}")
        continue
    districts = list(h[state].keys())[:3]
    for dist in districts:
        crops = h[state][dist][:5]
        print(f"{state} > {dist}: {crops}")
    print()

# Check potential_yields keys for Ludhiana wheat
with open(r"C:\Users\madha\OneDrive\Desktop\yeild estimator\output\potential_yields.json") as f:
    pot = json.load(f)

ludhiana_wheat = [k for k in pot if k.startswith("Ludhiana|") and "heat" in k.lower()]
print(f"Ludhiana wheat entries: {ludhiana_wheat}")

# What does hierarchy say for Ludhiana?
punjab_dist = list(h.get("Punjab", {}).keys())
if "Ludhiana" in punjab_dist:
    print(f"Hierarchy Ludhiana crops: {h['Punjab']['Ludhiana'][:10]}")
