import pandas as pd
import json

CSV = r"C:\Users\madha\OneDrive\Desktop\yeild estimator\output\master_data.csv"
WATER = r"C:\Users\madha\OneDrive\Desktop\yeild estimator\output\water_data.json"

df = pd.read_csv(CSV, usecols=["District_Name", "Yield_kg_ha", "Historical_Rainfall_mm"], low_memory=False)
df = df.dropna(subset=["Yield_kg_ha", "Historical_Rainfall_mm"])

# Historical mean rainfall per district
hist = df.groupby("District_Name")["Historical_Rainfall_mm"].mean().to_dict()

with open(WATER) as f:
    water = json.load(f)

# Update each district: add hist_precip and hist_water_factor
for name, wd in water.items():
    if name in hist:
        hist_mm = round(hist[name], 1)
        wd["hist_precip_mm"] = hist_mm
        et0 = wd.get("et0_mm", 1)
        wd["hist_water_factor"] = round(min(1.0, hist_mm / et0), 4) if et0 > 0 else 0.5
    else:
        wd["hist_precip_mm"] = wd.get("precip_mm", 0)
        wd["hist_water_factor"] = wd.get("water_factor", 0.5)

with open(WATER, "w") as f:
    json.dump(water, f)

# Show comparisons
for name in ["Bagalkot", "Ludhiana", "Jhunjhunu", "Idukki", "Pune"]:
    if name in water:
        w = water[name]
        print(f"  {name}: 2023={w['precip_mm']:.0f}mm factor={w['water_factor']:.3f}  hist={w['hist_precip_mm']:.0f}mm factor={w['hist_water_factor']:.3f}")

print(f"Updated {len(water)} districts")
