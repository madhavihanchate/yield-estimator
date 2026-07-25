"""Fast backtest - load CSV once, cache everything"""
import json, os, math
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

with open(os.path.join(OUTPUT_DIR, "potential_yields.json")) as f:
    potential = json.load(f)
with open(os.path.join(OUTPUT_DIR, "water_data.json")) as f:
    water_data = json.load(f)
with open(os.path.join(OUTPUT_DIR, "irrigation_data.json")) as f:
    irrigation = json.load(f)

master = pd.read_csv(os.path.join(OUTPUT_DIR, "master_data.csv"))
master["District_Name"] = master["District_Name"].str.strip()
master["Crop_Name"] = master["Crop_Name"].str.strip()

actual_2023 = master[master["Year"] == 2023].groupby(["District_Name", "Crop_Name"])["Yield_kg_ha"].mean().reset_index()
actual_2023 = actual_2023[actual_2023["Yield_kg_ha"] > 0]

# Pre-compute water factors
water_cache = {}
for dist in actual_2023["District_Name"].unique():
    wd = water_data.get(dist, {})
    hw = wd.get("hist_water_factor", 0.5)
    irr_s = irrigation.get(dist, {}).get("irrigation_score", 0.0)
    water_cache[dist] = min(1.0, hw + irr_s * (1.0 - hw))

# Pre-compute potential + water for matching pairs
pairs = []
for _, row in actual_2023.iterrows():
    dist, crop = row["District_Name"], row["Crop_Name"]
    pot_key = f"{dist}|{crop}"
    if pot_key in potential:
        pairs.append((potential[pot_key]["mean"], water_cache.get(dist, 0.5), row["Yield_kg_ha"]))

print(f"Matching pairs: {len(pairs)}")

def eval_model(soil_f, ndvi_f, label):
    preds = [p[0] * soil_f * p[1] * ndvi_f for p in pairs]
    acts = [p[2] for p in pairs]
    n = len(preds)
    errors = [a - pr for a, pr in zip(acts, preds)]
    rmse = math.sqrt(sum(e**2 for e in errors) / n)
    mae = sum(abs(e) for e in errors) / n
    mape = sum(abs(e/a)*100 for e,a in zip(errors, acts) if a > 0) / n
    mean_a = sum(acts) / n
    ss_res = sum(e**2 for e in errors)
    ss_tot = sum((a - mean_a)**2 for a in acts)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    w20 = sum(1 for e,a in zip(errors,acts) if abs(e/a)<0.20 if a>0)/n*100
    w30 = sum(1 for e,a in zip(errors,acts) if abs(e/a)<0.30 if a>0)/n*100
    w50 = sum(1 for e,a in zip(errors,acts) if abs(e/a)<0.50 if a>0)/n*100
    print(f"  {label:35s} RMSE={rmse:8.0f}  MAPE={mape:5.1f}%  R2={r2:.3f}  <20%={w20:.0f}%  <30%={w30:.0f}%  <50%={w50:.0f}%")

print("\n=== NDVI FACTOR SENSITIVITY (soil=0.67) ===")
for nf in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5]:
    eval_model(0.67, nf, f"soil=0.67 ndvi={nf}")

print("\n=== SOIL FACTOR SENSITIVITY (ndvi=1.0) ===")
for sf in [1.0, 0.9, 0.8, 0.7, 0.67, 0.6, 0.5, 0.4]:
    eval_model(sf, 1.0, f"soil={sf} ndvi=1.0")

print("\n=== BEST COMBINATION ===")
best_mape = 999
best_sf = 0
best_nf = 0
for sf in [x/100 for x in range(40, 105, 5)]:
    for nf in [x/100 for x in range(50, 105, 5)]:
        preds = [p[0] * sf * p[1] * nf for p in pairs]
        n = len(preds)
        mape = sum(abs((a-pr)/a)*100 for pr,a in zip(preds, [p[2] for p in pairs]) if a > 0) / n
        if mape < best_mape:
            best_mape = mape
            best_sf = sf
            best_nf = nf

eval_model(best_sf, best_nf, f"BEST: soil={best_sf} ndvi={best_nf}")

# Baseline: just potential
preds_b = [p[0] for p in pairs]
acts_b = [p[2] for p in pairs]
n = len(preds_b)
errors_b = [a-p for a,p in zip(acts_b, preds_b)]
rmse_b = math.sqrt(sum(e**2 for e in errors_b)/n)
mape_b = sum(abs(e/a)*100 for e,a in zip(errors_b, acts_b) if a>0)/n
mean_a = sum(acts_b)/n
ss_res = sum(e**2 for e in errors_b)
ss_tot = sum((a-mean_a)**2 for a in acts_b)
r2_b = 1 - ss_res/ss_tot
w50_b = sum(1 for e,a in zip(errors_b,acts_b) if abs(e/a)<0.50 if a>0)/n*100
print(f"\n  {'BASELINE: potential only':35s} RMSE={rmse_b:8.0f}  MAPE={mape_b:5.1f}%  R2={r2_b:.3f}  <50%={w50_b:.0f}%")
