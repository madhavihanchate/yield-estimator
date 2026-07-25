"""
Fetch external data for yield gap model:
1. Water data from Open-Meteo (precipitation + ET0) per district centroid
2. Vegetation proxy from NASA POWER (solar radiation + root zone wetness) per district centroid
Caches results. Re-runs skip already-fetched districts.
"""
import json
import os
import time
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CENTROIDS_PATH = os.path.join(OUTPUT_DIR, "centroids.json")
WATER_PATH = os.path.join(OUTPUT_DIR, "water_data.json")
NDVI_PATH = os.path.join(OUTPUT_DIR, "ndvi_data.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CENTROIDS_PATH, "r") as f:
    centroids = json.load(f)

valid = {k: v for k, v in centroids.items() if v.get("lat") is not None}
print(f"Valid centroids: {len(valid)} / {len(centroids)}")

# --- Water data (Open-Meteo) ---
water_cache = {}
if os.path.exists(WATER_PATH):
    with open(WATER_PATH, "r") as f:
        water_cache = json.load(f)
    print(f"Water cache: {len(water_cache)}")

to_fetch_water = [d for d in valid if d not in water_cache]
print(f"Water to fetch: {len(to_fetch_water)}")

session = requests.Session()
for i, district in enumerate(to_fetch_water):
    lat, lon = valid[district]["lat"], valid[district]["lon"]
    try:
        r = session.get("https://archive-api.open-meteo.com/v1/archive", params={
            "latitude": lat, "longitude": lon,
            "start_date": "2023-01-01", "end_date": "2023-12-31",
            "daily": "precipitation_sum,et0_fao_evapotranspiration",
            "timezone": "Asia/Kolkata",
        }, timeout=15)
        d = r.json().get("daily", {})
        precip = sum(v for v in d.get("precipitation_sum", []) if isinstance(v, (int, float)))
        et0 = sum(v for v in d.get("et0_fao_evapotranspiration", []) if isinstance(v, (int, float)))
        water_cache[district] = {
            "lat": lat, "lon": lon,
            "precip_mm": round(precip, 1),
            "et0_mm": round(et0, 1),
            "water_factor": round(min(1.0, precip / et0), 4) if et0 > 0 else 0.5,
        }
    except Exception as e:
        water_cache[district] = {"lat": lat, "lon": lon, "precip_mm": 0, "et0_mm": 0, "water_factor": 0.5, "error": str(e)}

    if (i + 1) % 50 == 0 or i == len(to_fetch_water) - 1:
        with open(WATER_PATH, "w") as f:
            json.dump(water_cache, f)
        print(f"  Water [{i+1}/{len(to_fetch_water)}]")

    time.sleep(0.5)

with open(WATER_PATH, "w") as f:
    json.dump(water_cache, f)
print(f"Water done: {len(water_cache)} districts")

# --- Vegetation proxy (NASA POWER) ---
ndvi_cache = {}
if os.path.exists(NDVI_PATH):
    with open(NDVI_PATH, "r") as f:
        ndvi_cache = json.load(f)
    print(f"NDVI cache: {len(ndvi_cache)}")

to_fetch_ndvi = [d for d in valid if d not in ndvi_cache]
print(f"NDVI to fetch: {len(to_fetch_ndvi)}")

for i, district in enumerate(to_fetch_ndvi):
    lat, lon = valid[district]["lat"], valid[district]["lon"]
    try:
        r = session.get("https://power.larc.nasa.gov/api/temporal/daily/point", params={
            "parameters": "ALLSKY_SFC_SW_DWN,GWETROOT,T2M,PRECTOTCORR",
            "community": "AG",
            "longitude": lon, "latitude": lat,
            "start": "20230601", "end": "20230930",
            "format": "JSON",
        }, timeout=15)
        data = r.json().get("properties", {}).get("parameter", {})

        def clean(vals):
            return [v for v in vals.values() if isinstance(v, (int, float)) and v > -900]

        rad = clean(data.get("ALLSKY_SFC_SW_DWN", {}))
        wet = clean(data.get("GWETROOT", {}))
        temp = clean(data.get("T2M", {}))

        mean_rad = sum(rad) / len(rad) if rad else 15.0
        mean_wet = sum(wet) / len(wet) if wet else 0.5
        mean_temp = sum(temp) / len(temp) if temp else 28.0

        rad_factor = min(1.0, mean_rad / 22.0)
        temp_factor = max(0.0, 1.0 - abs(mean_temp - 27.0) / 18.0)
        ndvi_factor = round(min(1.0, 0.5 * rad_factor + 0.5 * mean_wet + 0.2 * temp_factor), 4)

        ndvi_cache[district] = {
            "lat": lat, "lon": lon,
            "mean_radiation": round(mean_rad, 3),
            "mean_wetness": round(mean_wet, 4),
            "mean_temp": round(mean_temp, 2),
            "ndvi_factor": ndvi_factor,
        }
    except Exception as e:
        ndvi_cache[district] = {"lat": lat, "lon": lon, "ndvi_factor": 0.7, "error": str(e)}

    if (i + 1) % 50 == 0 or i == len(to_fetch_ndvi) - 1:
        with open(NDVI_PATH, "w") as f:
            json.dump(ndvi_cache, f)
        print(f"  NDVI [{i+1}/{len(to_fetch_ndvi)}]")

    time.sleep(0.5)

with open(NDVI_PATH, "w") as f:
    json.dump(ndvi_cache, f)
print(f"NDVI done: {len(ndvi_cache)} districts")
print("\nAll external data fetched.")
