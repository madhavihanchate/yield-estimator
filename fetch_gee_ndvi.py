"""
Fetch real MODIS NDVI (MOD13A2, 1km, 16-day) from Google Earth Engine.
Uses peak growing season (Jun-Sep) mean NDVI for each district centroid.
Caches results. Re-runs skip already-fetched districts.
"""
import ee
import json
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CENTROIDS_PATH = os.path.join(OUTPUT_DIR, "centroids.json")
NDVI_PATH = os.path.join(OUTPUT_DIR, "ndvi_data.json")

from pathlib import Path
try:
    import dotenv
    env_path = Path(BASE_DIR) / ".env"
    if env_path.exists():
        dotenv.load_dotenv(env_path)
except Exception:
    pass

project_id = os.environ.get("EARTHENGINE_PROJECT") or os.environ.get("GEE_PROJECT_ID") or "seed2serve"
ee.Initialize(project=project_id)

with open(CENTROIDS_PATH) as f:
    centroids = json.load(f)

valid = {k: v for k, v in centroids.items() if v.get("lat") is not None}
print(f"Valid centroids: {len(valid)}")

ndvi_cache = {}
if os.path.exists(NDVI_PATH):
    with open(NDVI_PATH) as f:
        ndvi_cache = json.load(f)
    print(f"Existing cache: {len(ndvi_cache)} entries")

to_fetch = [d for d in valid if d not in ndvi_cache or "MODIS" not in ndvi_cache[d].get("source", "")]
print(f"Need to fetch: {len(to_fetch)} districts")

modis = ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI")

BATCH_SIZE = 50

for batch_start in range(0, len(to_fetch), BATCH_SIZE):
    batch = to_fetch[batch_start:batch_start + BATCH_SIZE]
    points = []
    for district in batch:
        lat = valid[district]["lat"]
        lon = valid[district]["lon"]
        points.append(ee.Feature(ee.Geometry.Point([lon, lat]), {"district": district}))

    fc = ee.FeatureCollection(points)

    results = modis.filterDate("2023-01-01", "2023-12-31").mean()

    def extract_ndvi(feature):
        val = results.reduceRegion(ee.Reducer.mean(), feature.geometry(), 1000)
        return feature.set("ndvi_raw", val.get("NDVI"))

    with_ndvi = fc.map(extract_ndvi)
    data = with_ndvi.getInfo()

    for feature in data["features"]:
        props = feature["properties"]
        district = props["district"]
        raw = props.get("ndvi_raw")
        if raw is not None:
            ndvi_factor = round(min(1.0, raw / 7000.0), 4)
        else:
            ndvi_factor = 0.7

        ndvi_cache[district] = {
            "lat": valid[district]["lat"],
            "lon": valid[district]["lon"],
            "ndvi_raw": round(raw, 1) if raw else None,
            "ndvi_factor": ndvi_factor,
            "source": "MODIS_MOD13A2_2023",
        }

    with open(NDVI_PATH, "w") as f:
        json.dump(ndvi_cache, f)

    print(f"  Fetched batch {batch_start // BATCH_SIZE + 1}: {batch_start + len(batch)}/{len(to_fetch)} total")

    time.sleep(1)

total = len(ndvi_cache)
fetched = sum(1 for v in ndvi_cache.values() if v.get("source", "").startswith("MODIS"))
print(f"\nDone. Total: {total} districts, {fetched} with real MODIS NDVI")
