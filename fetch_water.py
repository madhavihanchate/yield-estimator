"""
Fetch precipitation + ET0 from Open-Meteo for each district centroid.
Free, no API key. Rate limit: be polite (0.5s between calls).
Caches results to output/water_data.json.
"""
import json
import os
import time
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CENTROIDS_PATH = os.path.join(OUTPUT_DIR, "centroids.json")
WATER_PATH = os.path.join(OUTPUT_DIR, "water_data.json")

with open(CENTROIDS_PATH, "r") as f:
    centroids = json.load(f)

existing = {}
if os.path.exists(WATER_PATH):
    with open(WATER_PATH, "r") as f:
        existing = json.load(f)
    print(f"Loaded {len(existing)} cached water records")

to_fetch = [d for d, v in centroids.items() if v.get("lat") is not None and d not in existing]
print(f"Need to fetch water data for: {len(to_fetch)} districts")

session = requests.Session()

for i, district in enumerate(to_fetch):
    lat = centroids[district]["lat"]
    lon = centroids[district]["lon"]

    try:
        resp = session.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "daily": "precipitation_sum,et0_fao_evapotranspiration",
                "timezone": "Asia/Kolkata",
            },
            timeout=15,
        )
        data = resp.json()
        daily = data.get("daily", {})
        precip_vals = [v for v in daily.get("precipitation_sum", []) if v is not None]
        et0_vals = [v for v in daily.get("et0_fao_evapotranspiration", []) if v is not None]

        total_precip = round(sum(precip_vals), 1) if precip_vals else 0
        total_et0 = round(sum(et0_vals), 1) if et0_vals else 0
        water_factor = round(min(1.0, total_precip / total_et0), 4) if total_et0 > 0 else 0.5

        existing[district] = {
            "lat": lat, "lon": lon,
            "precip_mm": total_precip,
            "et0_mm": total_et0,
            "water_factor": water_factor,
        }
    except Exception as e:
        existing[district] = {
            "lat": lat, "lon": lon,
            "precip_mm": 0, "et0_mm": 0,
            "water_factor": 0.5, "error": str(e),
        }

    if (i + 1) % 50 == 0 or i == len(to_fetch) - 1:
        with open(WATER_PATH, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  [{i+1}/{len(to_fetch)}] cached={len(existing)}")

    time.sleep(0.5)

with open(WATER_PATH, "w") as f:
    json.dump(existing, f, indent=2)
print(f"\nDone. Total water records: {len(existing)}")
