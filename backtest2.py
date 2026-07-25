"""Find optimal NDVI factor and test model variants"""
import json, os, math
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

with open(os.path.join(OUTPUT_DIR, "potential_yields.json")) as f:
    potential = json.load(f)
with open(os.path.join(OUTPUT_DIR, "water_data.json")) as f:
    water = json.load(f)
with open(os.path.join(OUTPUT_DIR, "irrigation_data.json")) as f:
    irrigation = json.load(f)

master = pd.read_csv(os.path.join(OUTPUT_DIR, "master_data.csv"))
master["District_Name"] = master["District_Name"].str.strip()
master["Crop_Name"] = master["Crop_Name"].str.strip()
actual_2023 = master[master["Year"] == 2023].groupby(["District_Name", "Crop_Name"])["Yield_kg_ha"].mean().reset_index()
actual_2023 = actual_2023[actual_2023["Yield_kg_ha"] > 0]

def soil_score(soc, pH, N, P, K):
    s_soc = min(25, (soc / 1.0) * 25)
    s_pH = max(0, 25 - abs(pH - 7.0) * 8)
    s_N = min(20, (N / 280.0) * 20)
    s_P = min(15, (P / 40.0) * 15)
    s_K = min(15, (K / 200.0) * 15)
    return (s_soc + s_pH + s_N + s_P + s_K) / 100.0

soil_f = soil_score(0.5, 6.8, 200, 20, 150)

def eval_model(ndvi_f, label):
    preds, acts = [], []
    for _, row in actual_2023.iterrows():
        dist, crop = row["District_Name"], row["Crop_Name"]
        pot_key = f"{dist}|{crop}"
        if pot_key not in potential: continue
        pot = potential[pot_key]["mean"]
        wd = water.get(dist, {})
        hw = wd.get("hist_water_factor", 0.5)
        irr_s = irrigation.get(dist, {}).get("irrigation_score", 0.0)
        wf = min(1.0, hw + irr_s * (1.0 - hw))
        pred = pot * soil_f * wf * ndvi_f
        preds.append(pred)
        acts.append(row["Yield_kg_ha"])

    n = len(preds)
    if n == 0: return
    errors = [a - p for a, p in zip(acts, preds)]
    rmse = math.sqrt(sum(e**2 for e in errors) / n)
    mae = sum(abs(e) for e in errors) / n
    mape = sum(abs(e/a)*100 for e,a in zip(errors, acts) if a>0) / n
    ss_res = sum(e**2 for e in errors)
    ss_tot = sum((a - sum(acts)/n)**2 for a in acts)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    within_20 = sum(1 for e,a in zip(errors,acts) if abs(e/a)<0.20 if a>0)/n*100
    within_30 = sum(1 for e,a in zip(errors,acts) if abs(e/a)<0.30 if a>0)/n*100
    within_50 = sum(1 for e,a in zip(errors,acts) if abs(e/a)<0.50 if a>0)/n*100
    print(f"  {label:30s} n={n:5d}  RMSE={rmse:8.0f}  MAE={mae:8.0f}  MAPE={mape:5.1f}%  R2={r2:.3f}  within20={within_20:.1f}%  within30={within_30:.1f}%  within50={within_50:.1f}%")

print("=== MODEL VARIANTS (all with soil + water, default params) ===")
eval_model(1.0,  "No NDVI (factor=1.0)")
eval_model(0.9,  "NDVI factor=0.9")
eval_model(0.85, "NDVI factor=0.85")
eval_model(0.8,  "NDVI factor=0.8")
eval_model(0.75, "NDVI factor=0.75")
eval_model(0.7,  "NDVI factor=0.7 (current)")
eval_model(0.6,  "NDVI factor=0.6")
eval_model(0.5,  "NDVI factor=0.5")

# Also test: potential only (no factors)
preds_p, acts_p = [], []
for _, row in actual_2023.iterrows():
    dist, crop = row["District_Name"], row["Crop_Name"]
    pot_key = f"{dist}|{crop}"
    if pot_key not in potential: continue
    preds_p.append(potential[pot_key]["mean"])
    acts_p.append(row["Yield_kg_ha"])
n = len(preds_p)
errors = [a-p for a,p in zip(acts_p, preds_p)]
rmse = math.sqrt(sum(e**2 for e in errors)/n)
mape = sum(abs(e/a)*100 for e,a in zip(errors, acts_p) if a>0)/n
ss_res = sum(e**2 for e in errors)
ss_tot = sum((a - sum(acts_p)/n)**2 for a in acts_p)
r2 = 1 - ss_res/ss_tot
print(f"\n  {'Potential only (no factors)':30s} n={n:5d}  RMSE={rmse:8.0f}  MAPE={mape:5.1f}%  R2={r2:.3f}")

# Optimal soil factor test
print("\n=== SOIL FACTOR SENSITIVITY ===")
for sf in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    preds, acts = [], []
    for _, row in actual_2023.iterrows():
        dist, crop = row["District_Name"], row["Crop_Name"]
        pot_key = f"{dist}|{crop}"
        if pot_key not in potential: continue
        pot = potential[pot_key]["mean"]
        wd = water.get(dist, {})
        hw = wd.get("hist_water_factor", 0.5)
        irr_s = irrigation.get(dist, {}).get("irrigation_score", 0.0)
        wf = min(1.0, hw + irr_s * (1.0 - hw))
        pred = pot * sf * wf
        preds.append(pred)
        acts.append(row["Yield_kg_ha"])
    n = len(preds)
    errors = [a-p for a,p in zip(acts,preds)]
    rmse = math.sqrt(sum(e**2 for e in errors)/n)
    mape = sum(abs(e/a)*100 for e,a in zip(errors, acts) if a>0)/n
    ss_res = sum(e**2 for e in errors)
    ss_tot = sum((a - sum(acts)/n)**2 for a in acts)
    r2 = 1 - ss_res/ss_tot
    print(f"  soil={sf:.1f}  RMSE={rmse:8.0f}  MAPE={mape:5.1f}%  R2={r2:.3f}")
