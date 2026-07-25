import pandas as pd
import numpy as np
from numpy import clip
import json

CSV = r"C:\Users\madha\OneDrive\Desktop\yeild estimator\output\master_data.csv"
OUT = r"C:\Users\madha\OneDrive\Desktop\yeild estimator\output\irrigation_data.json"

df = pd.read_csv(CSV, usecols=["District_Name", "Yield_kg_ha", "Historical_Rainfall_mm"], low_memory=False)
df = df.dropna(subset=["Yield_kg_ha", "Historical_Rainfall_mm"])

results = []
for name, grp in df.groupby("District_Name"):
    if len(grp) < 30:
        continue
    cv = grp["Yield_kg_ha"].std() / grp["Yield_kg_ha"].mean()
    corr = float(grp["Yield_kg_ha"].corr(grp["Historical_Rainfall_mm"]))
    my = grp["Yield_kg_ha"].mean()
    mr = grp["Historical_Rainfall_mm"].mean()
    yemm = my / max(mr, 1)
    results.append({"d": name, "cv": cv, "corr": corr, "my": my, "mr": mr, "yemm": yemm})

rd = pd.DataFrame(results)

ypm = rd["yemm"]
cv = rd["cv"]
corr_raw = rd["corr"]

# Signal 1: yield_per_mm — high = irrigated
ypm_p10, ypm_p90 = ypm.quantile(0.1), ypm.quantile(0.9)
ypm_norm = clip((ypm - ypm_p10) / (ypm_p90 - ypm_p10), 0, 1)

# Signal 2: yield-rain correlation — LOW = irrigated (yield doesn't depend on rain)
corr_p10, corr_p90 = corr_raw.quantile(0.1), corr_raw.quantile(0.9)
corr_norm = clip(1.0 - (corr_raw - corr_p10) / (corr_p90 - corr_p10), 0, 1)

# Signal 3: CV — low = irrigated (but noisy, low weight)
cv_p10, cv_p90 = cv.quantile(0.1), cv.quantile(0.9)
cv_norm = clip(1.0 - (cv - cv_p10) / (cv_p90 - cv_p10), 0, 1)

# New weights: trust yield_per_mm and correlation more, CV less
irrigation = 0.50 * ypm_norm + 0.30 * corr_norm + 0.20 * cv_norm
irrigation = irrigation.round(3)
rd["irrigation_score"] = irrigation.values

out = {}
for _, r in rd.iterrows():
    out[r["d"]] = {
        "mean_yield": round(float(r["my"]), 1),
        "mean_rain": round(float(r["mr"]), 1),
        "yield_per_mm": round(float(r["yemm"]), 3),
        "cv": round(float(r["cv"]), 3),
        "rain_yield_corr": round(float(r["corr"]), 3),
        "irrigation_score": round(float(r["irrigation_score"]), 3),
    }

with open(OUT, "w") as f:
    json.dump(out, f)

for name in ["Bagalkot", "Jhunjhunu", "Ludhiana", "Idukki", "Pune", "Kamrup Metropolitan", "Guntur", "Indore"]:
    if name in out:
        d = out[name]
        print(f"  {name}: irr={d['irrigation_score']:.3f} y/mm={d['yield_per_mm']:.1f} corr={d['rain_yield_corr']:.3f}")

print(f"\nTotal: {len(out)} districts")
