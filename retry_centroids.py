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
print(f"Missing: {len(missing)} districts. Retrying with state context...")

session = requests.Session()

for i, district in enumerate(missing):
    state = centroids[district].get("admin1", "")
    query = f"{district}, India"
    try:
        resp = session.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": query, "count": 10, "language": "en", "format": "json"},
            timeout=10,
        )
        data = resp.json()
        results = data.get("results", [])
        if results:
            best = results[0]
            centroids[district] = {
                "lat": best["latitude"],
                "lon": best["longitude"],
                "country": best.get("country_code", ""),
                "admin1": best.get("admin1", ""),
                "retry": True,
            }
        else:
            pass
    except Exception:
        pass
    time.sleep(0.15)

still_missing = [k for k, v in centroids.items() if v.get("lat") is None]
print(f"After retry: still missing {len(still_missing)}")

with open(CACHE_PATH, "w") as f:
    json.dump(centroids, f, indent=2)

found = sum(1 for v in centroids.values() if v.get("lat") is not None)
print(f"Total found: {found}/{len(centroids)}")
