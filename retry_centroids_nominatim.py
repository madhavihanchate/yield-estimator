import json
import os
import time
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CACHE_PATH = os.path.join(OUTPUT_DIR, "centroids.json")

with open(CACHE_PATH, "r") as f:
    centroids = json.load(f)

missing = [k for k, v in centroids.items() if v.get("lat") is None]
print(f"Missing: {len(missing)} districts. Trying Nominatim...")

session = requests.Session()
session.headers["User-Agent"] = "YieldEstimator/1.0 (academic project)"

found_count = 0
for i, district in enumerate(missing):
    queries = [
        f"{district}, India",
        f"{district} district, India",
    ]
    for query in queries:
        try:
            resp = session.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": 3, "countrycodes": "in"},
                timeout=10,
            )
            results = resp.json()
            if results:
                best = results[0]
                centroids[district] = {
                    "lat": float(best["lat"]),
                    "lon": float(best["lon"]),
                    "source": "nominatim",
                }
                found_count += 1
                break
        except Exception:
            pass
        time.sleep(1.1)

    if (i + 1) % 20 == 0 or i == len(missing) - 1:
        with open(CACHE_PATH, "w") as f:
            json.dump(centroids, f, indent=2)
        total_found = sum(1 for v in centroids.values() if v.get("lat") is not None)
        print(f"  [{i+1}/{len(missing)}] nominatim_found={found_count} total={total_found}/{len(centroids)}")

    time.sleep(1.1)

with open(CACHE_PATH, "w") as f:
    json.dump(centroids, f, indent=2)

still_missing = [k for k, v in centroids.items() if v.get("lat") is None]
total_found = sum(1 for v in centroids.values() if v.get("lat") is not None)
print(f"\nDone. Nominatim found: {found_count}. Still missing: {len(still_missing)}")
if still_missing:
    print(f"Missing: {still_missing}")
