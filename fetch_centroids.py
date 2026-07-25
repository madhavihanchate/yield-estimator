import requests
import json
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CSV_PATH = os.path.join(OUTPUT_DIR, "master_data.csv")
CACHE_PATH = os.path.join(OUTPUT_DIR, "centroids.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

import pandas as pd
master = pd.read_csv(CSV_PATH, usecols=["District_Name", "_sheet"], low_memory=False)
districts = sorted(master["District_Name"].unique().tolist())
print(f"Total unique districts: {len(districts)}")

existing = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r") as f:
        existing = json.load(f)
    print(f"Loaded {len(existing)} cached centroids")

to_fetch = [d for d in districts if d not in existing]
print(f"Need to geocode: {len(to_fetch)} districts")

session = requests.Session()

for i, district in enumerate(to_fetch):
    try:
        resp = session.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": district, "count": 5, "language": "en", "format": "json"},
            timeout=10,
        )
        data = resp.json()
        results = data.get("results", [])
        if results:
            best = results[0]
            existing[district] = {
                "lat": best["latitude"],
                "lon": best["longitude"],
                "country": best.get("country_code", ""),
                "admin1": best.get("admin1", ""),
            }
        else:
            existing[district] = {"lat": None, "lon": None, "error": "not_found"}
    except Exception as e:
        existing[district] = {"lat": None, "lon": None, "error": str(e)}

    if (i + 1) % 50 == 0 or i == len(to_fetch) - 1:
        with open(CACHE_PATH, "w") as f:
            json.dump(existing, f, indent=2)
        found = sum(1 for v in existing.values() if v.get("lat") is not None)
        print(f"  [{i+1}/{len(to_fetch)}] found={found}/{len(existing)}")

    time.sleep(0.15)

with open(CACHE_PATH, "w") as f:
    json.dump(existing, f, indent=2)

found = sum(1 for v in existing.values() if v.get("lat") is not None)
missing = [k for k, v in existing.items() if v.get("lat") is None]
print(f"\nDone. Found: {found}/{len(existing)}")
if missing:
    print(f"Missing ({len(missing)}): {missing[:20]}")
