"""
Backtest: predict 2023 yields using historical data (1997-2022) and compare with actual 2023.
Computes RMSE, MAE, R², and accuracy bands.
"""
import json
import os
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

with open(os.path.join(OUTPUT_DIR, "potential_yields.json")) as f:
    potential = json.load(f)
with open(os.path.join(OUTPUT_DIR, "water_data.json")) as f:
    water = json.load(f)
with open(os.path.join(OUTPUT_DIR, "irrigation_data.json")) as f:
    irrigation = json.load(f)
with open(os.path.join(OUTPUT_DIR, "centroids.json")) as f:
    centroids = json.load(f)

# Load raw data to get 2023 actuals
import pandas as pd
master = pd.read_csv(os.path.join(OUTPUT_DIR, "master_data.csv"))
master["District_Name"] = master["District_Name"].str.strip()
master["Crop_Name"] = master["Crop_Name"].str.strip()

# Get 2023 actual yields
actual_2023 = master[master["Year"] == 2023].groupby(["District_Name", "Crop_Name"])["Yield_kg_ha"].mean().reset_index()
actual_2023 = actual_2023[actual_2023["Yield_kg_ha"] > 0]

# Default soil (national average assumptions)
DEFAULT_SOC = 0.5
DEFAULT_PH = 6.8
DEFAULT_N = 200
DEFAULT_P = 20
DEFAULT_K = 150

def soil_score(soc, pH, N, P, K):
    s_soc = min(25, (soc / 1.0) * 25)
    s_pH = max(0, 25 - abs(pH - 7.0) * 8)
    s_N = min(20, (N / 280.0) * 20)
    s_P = min(15, (P / 40.0) * 15)
    s_K = min(15, (K / 200.0) * 15)
    return (s_soc + s_pH + s_N + s_P + s_K) / 100.0

soil_f = soil_score(DEFAULT_SOC, DEFAULT_PH, DEFAULT_N, DEFAULT_P, DEFAULT_K)

predictions = []
actuals = []

for _, row in actual_2023.iterrows():
    dist = row["District_Name"]
    crop = row["Crop_Name"]
    actual_yield = row["Yield_kg_ha"]
    pot_key = f"{dist}|{crop}"

    if pot_key not in potential:
        continue

    pot = potential[pot_key]["mean"]
    wd = water.get(dist, {})
    hist_water = wd.get("hist_water_factor", 0.5)
    irr = irrigation.get(dist, {})
    irr_score = irr.get("irrigation_score", 0.0)
    water_f = min(1.0, hist_water + irr_score * (1.0 - hist_water))

    # Use default NDVI since we're backtesting
    ndvi_f = 0.7

    predicted = pot * soil_f * water_f * ndvi_f
    predictions.append(predicted)
    actuals.append(actual_yield)

print(f"Backtest pairs: {len(predictions)}")
if not predictions:
    print("No matching pairs found!")
    exit()

# Compute metrics
errors = [a - p for a, p in zip(actuals, predictions)]
abs_errors = [abs(e) for e in errors]
sq_errors = [e**2 for e in errors]

mean_actual = sum(actuals) / len(actuals)
mean_predicted = sum(predictions) / len(predictions)

rmse = math.sqrt(sum(sq_errors) / len(sq_errors))
mae = sum(abs_errors) / len(abs_errors)
mape = sum(abs(e / a) * 100 for e, a in zip(errors, actuals) if a > 0) / len(actuals)

# R²
ss_res = sum(e**2 for e in errors)
ss_tot = sum((a - mean_actual)**2 for a in actuals)
r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

# Accuracy bands
within_10 = sum(1 for e, a in zip(errors, actuals) if abs(e / a) < 0.10 if a > 0) / len(actuals) * 100
within_20 = sum(1 for e, a in zip(errors, actuals) if abs(e / a) < 0.20 if a > 0) / len(actuals) * 100
within_30 = sum(1 for e, a in zip(errors, actuals) if abs(e / a) < 0.30 if a > 0) / len(actuals) * 100
within_50 = sum(1 for e, a in zip(errors, actuals) if abs(e / a) < 0.50 if a > 0) / len(actuals) * 100

print(f"\n=== ACCURACY METRICS ===")
print(f"Pairs tested: {len(predictions)}")
print(f"Mean actual yield: {mean_actual:.0f} kg/ha")
print(f"Mean predicted yield: {mean_predicted:.0f} kg/ha")
print(f"RMSE: {rmse:.0f} kg/ha")
print(f"MAE: {mae:.0f} kg/ha")
print(f"MAPE: {mape:.1f}%")
print(f"R-squared: {r2:.3f}")
print(f"\n=== ACCURACY BANDS ===")
print(f"Within 10% of actual: {within_10:.1f}%")
print(f"Within 20% of actual: {within_20:.1f}%")
print(f"Within 30% of actual: {within_30:.1f}%")
print(f"Within 50% of actual: {within_50:.1f}%")

# Top 10 most common crops
from collections import Counter
crop_counts = Counter(row["Crop_Name"] for _, row in actual_2023.iterrows() if f'{row["District_Name"]}|{row["Crop_Name"]}' in potential)
print(f"\n=== MOST TESTED CROPS ===")
for crop, count in crop_counts.most_common(10):
    crop_preds = []
    crop_actuals = []
    for _, row in actual_2023[actual_2023["Crop_Name"] == crop].iterrows():
        dist = row["District_Name"]
        pot_key = f"{dist}|{crop}"
        if pot_key in potential:
            pot = potential[pot_key]["mean"]
            wd = water.get(dist, {})
            hw = wd.get("hist_water_factor", 0.5)
            irr_s = irrigation.get(dist, {}).get("irrigation_score", 0.0)
            wf = min(1.0, hw + irr_s * (1.0 - hw))
            pred = pot * soil_f * wf * 0.7
            crop_preds.append(pred)
            crop_actuals.append(row["Yield_kg_ha"])
    if crop_preds:
        avg_err = sum(abs(a-p)/a*100 for a,p in zip(crop_actuals, crop_preds)) / len(crop_preds)
        print(f"  {crop}: {count} districts, avg error {avg_err:.0f}%")
