import json
import os
import logging
from django.http import JsonResponse
from django.shortcuts import render

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
CENTROIDS_PATH = os.path.join(OUTPUT_DIR, "centroids.json")

_potential = None
_water = None
_irrigation = None
_lookup = None
_categories = None
_hierarchy = None
_centroids = None

_gee_initialized = False
_gee_error = None

FAO_KC = {
    "Rice": 1.2, "Wheat": 1.15, "Maize": 1.2, "Corn": 1.2,
    "Cotton": 1.15, "Sugarcane": 1.25, "Groundnut": 1.15,
    "Soybean": 1.15, "Soyabean": 1.15, "Bajra (Pearl Millet)": 1.1,
    "Jowar (Sorghum)": 1.1, "Bajra": 1.1, "Jowar": 1.1,
    "Ragi (Finger Millet)": 1.1, "Ragi": 1.1,
    "Pulses": 1.15, "Gram": 1.15, "Tur (Arhar)": 1.1,
    "Moong (Green Gram)": 1.1, "Urad (Black Gram)": 1.1,
    "Onion": 1.05, "Potato": 1.1, "Tomato": 1.15,
    "Banana": 1.15, "Mango": 0.85, "Grapes": 0.85,
    "Tea": 0.95, "Coffee": 0.95, "Rubber": 1.0,
    "Coconut": 0.9, "Arecanut": 0.9, "Betelvine": 0.9,
    "Black Pepper": 0.95, "Cardamom": 0.95,
    "Tobacco": 1.05, "Jute": 1.15, "Mesta": 1.15,
    "Linseed": 1.1, "Sesamum": 1.1, "Nigerseed": 1.0,
    "Castor": 1.1, "Rapeseed & Mustard": 1.1, "Rapeseed": 1.1, "Mustard": 1.1,
    "Safflower": 1.05, "Sunflower": 1.1,
    "Horsegram": 1.0, "Cowpea": 1.1, "Lentil": 1.1,
    "Barley": 1.1, "Oats": 1.1, "Rice (Paddy)": 1.2,
}
DEFAULT_KC = 1.1


def _init_gee():
    global _gee_initialized, _gee_error
    if _gee_initialized or _gee_error:
        return
    try:
        import ee
        ee.Initialize(project="yield-estimate-502708")
        _gee_initialized = True
        logger.info("GEE initialized successfully")
    except Exception as e:
        _gee_error = str(e)
        logger.warning("GEE init failed: %s", e)


def _fetch_ndvi_live(district):
    import ee
    cd = _centroids.get(district)
    if not cd or cd.get("lat") is None:
        return None
    point = ee.Geometry.Point([cd["lon"], cd["lat"]])
    modis = ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI")
    img = modis.filterDate("2023-01-01", "2023-12-31").mean()
    raw = img.reduceRegion(ee.Reducer.mean(), point, 1000).get("NDVI").getInfo()
    if raw is None:
        return None
    return {
        "ndvi_raw": round(raw, 1),
        "ndvi_factor": round(min(1.0, raw / 7000.0), 4),
        "source": "MODIS_MOD13A2_2023_live",
    }


def _load():
    global _potential, _water, _irrigation, _lookup, _categories, _hierarchy, _centroids
    if _potential is not None:
        return
    with open(os.path.join(OUTPUT_DIR, "potential_yields.json")) as f:
        _potential = json.load(f)
    with open(os.path.join(OUTPUT_DIR, "water_data.json")) as f:
        _water = json.load(f)
    with open(os.path.join(OUTPUT_DIR, "irrigation_data.json")) as f:
        _irrigation = json.load(f)
    with open(os.path.join(OUTPUT_DIR, "lookup.json")) as f:
        _lookup = json.load(f)
    with open(os.path.join(OUTPUT_DIR, "crop_categories.json")) as f:
        _categories = json.load(f)
    hpath = os.path.join(OUTPUT_DIR, "hierarchy.json")
    if os.path.exists(hpath):
        with open(hpath) as f:
            _hierarchy = json.load(f)
    if os.path.exists(CENTROIDS_PATH):
        with open(CENTROIDS_PATH) as f:
            _centroids = json.load(f)
    _init_gee()


def _soil_score(soc, pH, N, P, K):
    s_soc = min(25, (soc / 1.0) * 25)
    s_pH = max(0, 25 - abs(pH - 7.0) * 8)
    s_N = min(20, (N / 280.0) * 20)
    s_P = min(15, (P / 40.0) * 15)
    s_K = min(15, (K / 200.0) * 15)
    total = s_soc + s_pH + s_N + s_P + s_K
    if total >= 80:
        grade = "Excellent"
    elif total >= 65:
        grade = "Good"
    elif total >= 50:
        grade = "Fair"
    elif total >= 35:
        grade = "Poor"
    else:
        grade = "Critical"
    return {
        "total": round(total, 1),
        "soc": round(s_soc, 1),
        "pH": round(s_pH, 1),
        "N": round(s_N, 1),
        "P": round(s_P, 1),
        "K": round(s_K, 1),
        "grade": grade,
    }


def index(request):
    _load()
    states = sorted(_hierarchy.keys()) if _hierarchy else []
    return render(request, "index.html", {
        "states": states,
        "hierarchy_json": json.dumps(_hierarchy or {}),
    })


def api_gee_status(request):
    _load()
    return JsonResponse({
        "authenticated": _gee_initialized,
        "error": _gee_error,
    })


def api_predict(request):
    _load()
    district = request.GET.get("district", "").strip()
    crop = request.GET.get("crop", "").strip()
    soc = float(request.GET.get("soc", 0.5))
    pH = float(request.GET.get("pH", 6.5))
    N = float(request.GET.get("N", 200))
    P = float(request.GET.get("P", 20))
    K = float(request.GET.get("K", 150))

    if not district or not crop:
        return JsonResponse({"error": "District and crop are required."}, status=400)

    pot_key = f"{district}|{crop}"
    if pot_key not in _potential:
        district_matches = [d for d in _lookup["districts"] if d.lower() == district.lower()]
        crop_matches = [c for c in _lookup["crops"] if c.lower() == crop.lower()]
        msg = "No yield history found for this district-crop pair."
        if district_matches and crop_matches:
            msg = f"'{district_matches[0]}' and '{crop_matches[0]}' exist but have no shared data."
        return JsonResponse({"status": "not_available", "message": msg})

    pot = _potential[pot_key]
    potential_yield = pot["mean"]

    soil = _soil_score(soc, pH, N, P, K)
    soil_factor = soil["total"] / 100.0

    wd = _water.get(district, {})
    hist_water = wd.get("hist_water_factor", 0.5)
    raw_water = wd.get("water_factor", 0.5)

    irr = _irrigation.get(district, {})
    irr_score = irr.get("irrigation_score", 0.0)

    water_factor = round(min(1.0, hist_water + irr_score * (1.0 - hist_water)), 4)

    ndvi_data = None
    ndvi_factor = 0.7
    ndvi_raw = 0
    ndvi_source = ""

    if _gee_initialized:
        try:
            ndvi_data = _fetch_ndvi_live(district)
        except Exception as e:
            logger.warning("GEE NDVI fetch failed for %s: %s", district, e)

    if ndvi_data:
        ndvi_factor = ndvi_data["ndvi_factor"]
        ndvi_raw = ndvi_data["ndvi_raw"]
        ndvi_source = ndvi_data["source"]

    predicted = potential_yield * soil_factor * water_factor * ndvi_factor

    return JsonResponse({
        "status": "available",
        "data": {
            "district": district,
            "crop": crop,
            "potential_yield": round(potential_yield, 1),
            "soil": soil,
            "soil_factor": round(soil_factor, 4),
            "rain_water_factor": round(raw_water, 4),
            "hist_water_factor": round(hist_water, 4),
            "irrigation_score": round(irr_score, 3),
            "water_factor": round(water_factor, 4),
            "precip_mm": wd.get("precip_mm", 0),
            "hist_precip_mm": wd.get("hist_precip_mm", 0),
            "et0_mm": wd.get("et0_mm", 0),
            "ndvi_factor": round(ndvi_factor, 4),
            "ndvi_raw": ndvi_raw,
            "ndvi_source": ndvi_source,
            "gee_authenticated": _gee_initialized,
            "predicted_yield": round(predicted, 1),
            "n_years": pot["n"],
            "max_yield": pot["max"],
            "p90_yield": pot["p90"],
        },
    })


def api_lookup(request):
    _load()
    return JsonResponse(_lookup)
