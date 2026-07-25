"""
Compute potential yields from workbook: historical mean Yield_kg_ha per district-crop.
Also compute potential_yields_90pct for top-end estimates.
"""
import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CSV_PATH = os.path.join(OUTPUT_DIR, "master_data.csv")

df = pd.read_csv(CSV_PATH, usecols=["District_Name", "Crop_Name", "Yield_kg_ha"], low_memory=False)
df = df.dropna(subset=["Yield_kg_ha"])

grouped = df.groupby(["District_Name", "Crop_Name"])["Yield_kg_ha"]

potentials = grouped.agg(["mean", "median", "std", "count", "max", lambda x: x.quantile(0.9)]).reset_index()
potentials.columns = ["District_Name", "Crop_Name", "mean_yield", "median_yield", "std_yield", "n_years", "max_yield", "p90_yield"]

potentials["mean_yield"] = potentials["mean_yield"].round(1)
potentials["median_yield"] = potentials["median_yield"].round(1)
potentials["std_yield"] = potentials["std_yield"].round(1)
potentials["max_yield"] = potentials["max_yield"].round(1)
potentials["p90_yield"] = potentials["p90_yield"].round(1)

records = {}
for _, row in potentials.iterrows():
    key = f"{row['District_Name']}|{row['Crop_Name']}"
    records[key] = {
        "mean": row["mean_yield"],
        "median": row["median_yield"],
        "std": row["std_yield"],
        "max": row["max_yield"],
        "p90": row["p90_yield"],
        "n": int(row["n_years"]),
    }

out_path = os.path.join(OUTPUT_DIR, "potential_yields.json")
with open(out_path, "w") as f:
    json.dump(records, f)

print(f"Potential yields computed: {len(records)} district-crop pairs")
print(f"Saved to {out_path}")

# Also build the crop→category lookup
df2 = pd.read_csv(CSV_PATH, usecols=["Crop_Name", "Category"], low_memory=False)
crop_cat = dict(df2.drop_duplicates().values)
cat_path = os.path.join(OUTPUT_DIR, "crop_categories.json")
with open(cat_path, "w") as f:
    json.dump(crop_cat, f)
print(f"Crop categories: {len(crop_cat)}")
