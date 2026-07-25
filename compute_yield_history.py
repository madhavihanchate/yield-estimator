"""Generate yearly yield history per district-crop pair from master_data.csv"""
import json, os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

master = pd.read_csv(os.path.join(OUTPUT_DIR, "master_data.csv"))
master["District_Name"] = master["District_Name"].str.strip()
master["Crop_Name"] = master["Crop_Name"].str.strip()

# Only keep rows with valid yields
master = master[master["Yield_kg_ha"] > 0]

# Group by district, crop, year -> mean yield
yearly = master.groupby(["District_Name", "Crop_Name", "Year"]).agg(
    yield_kg_ha=("Yield_kg_ha", "mean"),
    area_ha=("Area_Hectares", "sum"),
    production_t=("Production_Tonnes", "sum"),
).reset_index()

# Build nested dict: { "District|Crop": { "2023": {yield, area, production}, ... } }
history = {}
for _, row in yearly.iterrows():
    key = f"{row['District_Name']}|{row['Crop_Name']}"
    yr = str(int(row["Year"]))
    if key not in history:
        history[key] = {}
    history[key][yr] = {
        "yield": round(row["yield_kg_ha"], 1),
        "area": round(row["area_ha"], 1),
        "production": round(row["production_t"], 1),
    }

out_path = os.path.join(OUTPUT_DIR, "yield_history.json")
with open(out_path, "w") as f:
    json.dump(history, f)

total_pairs = len(history)
total_entries = sum(len(v) for v in history.values())
print(f"Saved {total_pairs} district-crop pairs, {total_entries} yearly entries")
print(f"File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")
