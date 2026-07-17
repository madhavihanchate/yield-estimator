"""
Yield Estimator — simple dashboard
State + District + N/P/K/OC/pH + Crop -> histogram of historical yield + probable current-year estimate.

DATA PRIORITY (as specified):
  1. crop_yield.csv (state-level, 1997-2020) is ALWAYS checked first for the historical series baseline.
  2. apy_district.csv (parsed from the DES xls, district-level, 2020-21 to 2022-23, 34 mostly
     horticulture/commercial crops) is used ONLY to refine to district-level when it has matching
     non-null data for that state+district+crop. If not found there, we fall back to the
     state-level average from crop_yield.csv.

IMPORTANT CAVEAT (tell this to whoever reviews the demo):
  apy_district.csv does NOT include Rice and has essentially no data for Wheat in major wheat
  states — it's a horticulture/commercial-crop-heavy report. Most staple-crop queries will
  silently resolve to the state-level CSV, not the district-level file. This is expected
  behavior given the fallback rule, not a bug — but it means "district-level" is not guaranteed
  for every crop.

This is a PROBABLE ESTIMATE, not a validated forecaster:
  - Historical bars = real recorded yield (state or district average, whichever resolved).
  - "Current (Predicted)" bar = baseline historical average adjusted by a transparent heuristic
    (soil score + live rainfall deviation + live NDVI deviation from historical norms).
    It is NOT a trained/validated regression. Treat the current-year number as directional,
    not authoritative — say this explicitly in any submission/demo.
"""

import re
import requests
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

CSV_PATH = "data/crop_yield.csv"
XLS_PARSED_PATH = "data/apy_district.csv"

def _clean_num(x):
    if pd.isna(x) or x == "":
        return None
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None

def _strip_prefix(s):
    """Strip DES-style '12. ' numbering prefix from state/district names."""
    if not isinstance(s, str):
        return s
    return re.sub(r"^\d+\.\s*", "", s).strip()

def _norm(s):
    return str(s).strip().lower() if s is not None else ""

# --- national/state-level (priority 1) ---
csv_df = pd.read_csv(CSV_PATH)
csv_df["Crop"] = csv_df["Crop"].str.strip()
csv_df["State"] = csv_df["State"].str.strip()
csv_df["Season"] = csv_df["Season"].str.strip()
csv_df["_state_n"] = csv_df["State"].apply(_norm)
csv_df["_crop_n"] = csv_df["Crop"].apply(_norm)

# --- crop -> yield unit (from real xls headers; most Indian APY crops are Tonne/Hectare,
#     but Coconut is Nuts/Hectare and Cotton/Mesta are Bales/Hectare — don't assume) ---
units_df = pd.read_csv("data/crop_units.csv")
CROP_UNITS = {_norm(r["crop"]): r["unit"] for _, r in units_df.iterrows()}

# --- crop -> ideal soil ranges (from India_State_Wise_Crops.xlsx, agronomic literature) ---
import json
with open("data/crop_soil_ranges.json") as f:
    CROP_SOIL_RANGES = json.load(f)

# --- crop -> peak-season NDVI baseline (from remote sensing literature) ---
with open("data/crop_ndvi_baseline.json") as f:
    _ndvi_raw = json.load(f)
NDVI_BASELINE = {k: v for k, v in _ndvi_raw.items() if not k.startswith("_")}
NDVI_DEFAULT = _ndvi_raw.get("_default", 0.65)

def _find_soil_ranges(crop):
    """Find ideal soil ranges for a crop. Tries exact match, then fuzzy partial match."""
    c_n = _norm(crop)
    # Exact match
    if c_n in CROP_SOIL_RANGES:
        return CROP_SOIL_RANGES[c_n]
    # Try partial match (e.g. "Rice" matches "Rice (Paddy)")
    for key in CROP_SOIL_RANGES:
        if c_n in _norm(key) or _norm(key) in c_n:
            return CROP_SOIL_RANGES[key]
    return None

def _get_ndvi_baseline(crop):
    """Look up peak-season NDVI baseline for a crop. Fuzzy match, then default."""
    c_n = _norm(crop)
    if c_n in NDVI_BASELINE:
        return NDVI_BASELINE[c_n]
    for key in NDVI_BASELINE:
        if c_n in _norm(key) or _norm(key) in c_n:
            return NDVI_BASELINE[key]
    return NDVI_DEFAULT

def soil_range_score(n, p, k, oc, ph, ranges):
    """Score user's soil inputs against ideal ranges for a crop. Returns 0-100."""
    scores = []
    param_map = {
        'oc': oc, 'n': n, 'p': p, 'k': k, 'ph': ph
    }
    for param, val in param_map.items():
        if val is None or param not in ranges or ranges[param] is None:
            continue
        low, high = ranges[param]
        if low <= val <= high:
            scores.append(100)  # Perfect - within ideal range
        elif val < low:
            # Below range - score based on how far below
            deficit = (low - val) / max(low, 0.01)
            scores.append(max(0, 100 - deficit * 100))
        else:
            # Above range - score based on how far above
            excess = (val - high) / max(high, 0.01)
            scores.append(max(0, 100 - excess * 100))
    return round(sum(scores) / len(scores), 1) if scores else 50.0

def get_crop_unit(crop):
    # default assumption for crops outside the xls's 34 — most Indian APY yields
    # are reported in Tonne/Hectare, but this is an assumption, not verified per-crop.
    return CROP_UNITS.get(_norm(crop), "Tonne/Hectare (assumed)")
xls_df = pd.read_csv(XLS_PARSED_PATH)
xls_df["state"] = xls_df["state"].apply(_strip_prefix)
xls_df["district"] = xls_df["district"].apply(_strip_prefix)
xls_df["yield_num"] = xls_df["yield_num"] if "yield_num" in xls_df.columns else xls_df["yield"].apply(_clean_num)
xls_df["year_start"] = xls_df["year"].str.extract(r"(\d{4})").astype(float)
xls_df["_state_n"] = xls_df["state"].apply(_norm)
xls_df["_district_n"] = xls_df["district"].apply(_norm)
xls_df["_crop_n"] = xls_df["crop"].apply(_norm)


def get_historical_series(state, district, crop):
    """
    Returns (source, {year: yield}) where source is 'district' or 'state'.
    CSV (state) is priority 1 in the sense that it's what we KNOW covers this crop broadly;
    but per the agreed rule, we try district (xls) first and fall back to state (csv) average
    when district has no matching non-null rows for this state+district+crop.
    """
    s_n, d_n, c_n = _norm(state), _norm(district), _norm(crop)

    # try district-level first
    sub = xls_df[
        (xls_df["_state_n"] == s_n)
        & (xls_df["_district_n"] == d_n)
        & (xls_df["_crop_n"] == c_n)
        & (xls_df["yield_num"].notna())
    ]
    if len(sub) > 0:
        series = sub.groupby("year_start")["yield_num"].mean().to_dict()
        return "district (xls)", {int(k): v for k, v in series.items()}

    # fallback: state-level average from CSV
    sub2 = csv_df[(csv_df["_state_n"] == s_n) & (csv_df["_crop_n"] == c_n)]
    if len(sub2) > 0:
        series = sub2.groupby("Crop_Year")["Yield"].mean().to_dict()
        return "state (csv fallback)", {int(k): v for k, v in series.items()}

    return None, {}


def get_historical_rainfall_avg(state, crop):
    """Average annual rainfall for this state+crop from the CSV, as a baseline for the live comparison."""
    s_n, c_n = _norm(state), _norm(crop)
    sub = csv_df[(csv_df["_state_n"] == s_n) & (csv_df["_crop_n"] == c_n)]
    if len(sub) > 0:
        return sub["Annual_Rainfall"].mean()
    sub2 = csv_df[csv_df["_state_n"] == s_n]
    if len(sub2) > 0:
        return sub2["Annual_Rainfall"].mean()
    return None


# ---------------------------------------------------------------------------
# 2. Soil suitability score (from the agronomist's formula)
# ---------------------------------------------------------------------------

def soil_suitability_score(n, p, k, oc, ph):
    soc_pts = min(25, (oc / 3.0) * 25) if oc is not None else 0
    ph_pts = max(0, 25 - abs(ph - 6.5) * 8) if ph is not None else 0
    n_pts = min(20, (n / 300) * 20) if n is not None else 0
    p_pts = min(15, (p / 40) * 15) if p is not None else 0
    k_pts = min(15, (k / 200) * 15) if k is not None else 0
    total = soc_pts + ph_pts + n_pts + p_pts + k_pts
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
    return round(total, 1), grade


def _grade(score):
    if score >= 80:
        return "Excellent"
    elif score >= 65:
        return "Good"
    elif score >= 50:
        return "Fair"
    elif score >= 35:
        return "Poor"
    else:
        return "Critical"


# ---------------------------------------------------------------------------
# 3. Live weather (Open-Meteo — free, no key)
# ---------------------------------------------------------------------------

def geocode(district, state):
    """Primary: GEE FAO GAUL district centroid (exact match, same polygon used for NDVI).
    Fallback: Open-Meteo name search (city/place index — often misses formal district names)."""
    if _try_init_ee():
        try:
            import ee
            feats = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(
                ee.Filter.And(
                    ee.Filter.eq("ADM0_NAME", "India"),
                    ee.Filter.eq("ADM2_NAME", district.title()),
                )
            )
            if feats.size().getInfo() > 0:
                centroid = feats.geometry().centroid(1).coordinates().getInfo()
                return centroid[1], centroid[0]  # [lon, lat] -> (lat, lon)
        except Exception:
            pass

    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": district, "count": 10, "country": "IN"},
            timeout=8,
        )
        data = r.json()
        results = data.get("results", [])
        if not results:
            return None, None
        state_n = _norm(state)
        for res in results:
            if _norm(res.get("admin1", "")) == state_n:
                return res["latitude"], res["longitude"]
        return results[0]["latitude"], results[0]["longitude"]
    except Exception:
        pass
    return None, None


def get_current_rainfall(lat, lon, year):
    """Cumulative rainfall for the current year to date, via Open-Meteo Archive API.
    End date is capped at a few days before today — the Archive API has no data for
    future or very recent dates, and requesting past 'today' was silently failing."""
    import datetime
    today = datetime.date.today()
    end_date = min(datetime.date(year, 12, 31), today - datetime.timedelta(days=3))
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": f"{year}-01-01",
                "end_date": end_date.isoformat(),
                "daily": "precipitation_sum",
                "timezone": "auto",
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        vals = data.get("daily", {}).get("precipitation_sum", [])
        vals = [v for v in vals if v is not None]
        return sum(vals) if vals else None
    except Exception as e:
        print(f"[rainfall fetch failed] {e}")  # visible in your terminal for debugging
        return None


# ---------------------------------------------------------------------------
# 4. Live NDVI (Google Earth Engine — requires YOUR OWN service-account credentials)
# ---------------------------------------------------------------------------
# NOTE: this needs `earthengine-api` installed AND a GEE service account key configured
# via the EE_SERVICE_ACCOUNT / EE_KEY_FILE env vars (see README below). If not configured,
# this fails gracefully and the app uses a neutral NDVI factor instead of crashing.

_ee_initialized = False
_EE_PROJECT = "yield-estimate-502708"  # verified working — see conversation notes

def _try_init_ee():
    global _ee_initialized
    if _ee_initialized:
        return True
    try:
        import ee
        ee.Initialize(project=_EE_PROJECT)
        _ee_initialized = True
        return True
    except Exception:
        return False


def get_current_ndvi(state, district):
    """Latest available MODIS NDVI composite for this district, via GEE. Returns None on any failure."""
    if not _try_init_ee():
        return None
    try:
        import ee
        districts = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(
            ee.Filter.And(
                ee.Filter.eq("ADM0_NAME", "India"),
                ee.Filter.eq("ADM2_NAME", district.title()),
            )
        )
        if districts.size().getInfo() == 0:
            return None
        latest = ee.ImageCollection("MODIS/061/MOD13Q1").sort("system:time_start", False).first()
        stat = latest.select("NDVI").reduceRegion(
            reducer=ee.Reducer.mean(), geometry=districts.geometry(), scale=250, maxPixels=1e9
        )
        val = stat.getInfo().get("NDVI")
        return (val / 10000) if val is not None else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 5. Prediction heuristic (transparent, not a trained model — see module docstring)
# ---------------------------------------------------------------------------

def predict_current_yield(historical_series, soil_score, rainfall_factor, ndvi_factor):
    if not historical_series:
        return None
    baseline = sum(historical_series.values()) / len(historical_series)

    soil_factor = soil_score / 65.0          # 65 = "Good" grade threshold, used as neutral point
    soil_factor = max(0.5, min(1.5, soil_factor))

    rf = rainfall_factor if rainfall_factor is not None else 1.0
    rf = max(0.5, min(1.5, rf))

    nf = ndvi_factor if ndvi_factor is not None else 1.0
    nf = max(0.5, min(1.5, nf))

    combined = 0.4 * soil_factor + 0.3 * rf + 0.3 * nf
    return round(baseline * combined, 3)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

get_unit = get_crop_unit  # alias — response building below calls get_unit()


@app.route("/")
def index():
    states = sorted(csv_df["State"].unique().tolist())
    crops = sorted(csv_df["Crop"].unique().tolist())
    return render_template("index.html", states=states, crops=crops)


@app.route("/api/districts")
def api_districts():
    state = request.args.get("state", "").strip()
    s_n = _norm(state)
    districts = sorted(xls_df[xls_df["_state_n"] == s_n]["district"].dropna().unique().tolist())
    has_csv = _norm(state) in csv_df["_state_n"].values
    return jsonify({
        "districts": districts,
        "csv_available": has_csv,
        "note": "" if districts else "No district-level data — will use state-level average from CSV" if has_csv else "No data available for this state"
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(silent=True) or {}
    state = (body.get("state") or "").strip()
    district = (body.get("district") or "").strip()
    crop = (body.get("crop") or "").strip()
    n = _clean_num(body.get("n"))
    p = _clean_num(body.get("p"))
    k = _clean_num(body.get("k"))
    oc = _clean_num(body.get("oc"))
    ph = _clean_num(body.get("ph"))

    errors = []
    if not state:
        errors.append("State is required.")
    if not crop:
        errors.append("Crop is required.")
    if n is None and p is None and k is None and oc is None and ph is None:
        errors.append("At least one soil parameter (N, P, K, OC, or pH) is required.")
    if errors:
        return jsonify({"error": " ".join(errors)}), 400

    source, series = get_historical_series(state, district, crop)

    soil_score, soil_grade = soil_suitability_score(n, p, k, oc, ph)

    lat, lon = geocode(district, state)
    current_year = pd.Timestamp.now().year
    rainfall_factor = None
    ndvi_factor = None
    ndvi_note = "unavailable (GEE not configured or district not matched) — used neutral value"

    if lat is not None:
        current_rain = get_current_rainfall(lat, lon, current_year)
        hist_rain = get_historical_rainfall_avg(state, crop)
        if current_rain is not None and pd.notna(hist_rain) and hist_rain > 0:
            rainfall_factor = current_rain / hist_rain

        ndvi_val = get_current_ndvi(state, district)
        if ndvi_val is not None:
            # Compare live NDVI against crop-specific peak-season baseline from literature.
            ndvi_baseline = _get_ndvi_baseline(crop)
            ndvi_factor = ndvi_val / ndvi_baseline
            ndvi_note = f"live NDVI={round(ndvi_val,3)} vs {crop} baseline {ndvi_baseline}"

    if not series:
        # Crop not in CSV dataset — try soil-range-based estimate from agronomic data
        ranges = _find_soil_ranges(crop)
        if ranges:
            soil_score_new = soil_range_score(n, p, k, oc, ph, ranges)
            # Estimate yield from soil score + weather factors
            rf = rainfall_factor if rainfall_factor is not None else 1.0
            nf = ndvi_factor if ndvi_factor is not None else 1.0
            # Typical Indian crop yield range: 0.5–8 t/ha depending on crop
            # Use a middle reference (2.5 t/ha) scaled by soil score and weather
            estimated_yield = round(2.5 * (soil_score_new / 100.0) * rf * nf, 2)

            return jsonify({
                "mode": "suitability_estimate",
                "note": f"No historical yield data for '{crop}' — estimated from agronomic soil ranges + live weather. Treat as approximate.",
                "soil_score": soil_score_new,
                "soil_grade": _grade(soil_score_new),
                "ideal_ranges": ranges,
                "rainfall_factor": rainfall_factor,
                "ndvi_factor": ndvi_factor,
                "ndvi_note": ndvi_note,
                "estimated_yield": estimated_yield,
                "unit": get_unit(crop),
            })

        # No soil ranges found either — pure suitability indicator only
        rf = rainfall_factor if rainfall_factor is not None else 1.0
        nf = ndvi_factor if ndvi_factor is not None else 1.0
        combined = 0.5 * max(0.5, min(1.5, rf)) + 0.5 * max(0.5, min(1.5, nf))
        suitability = round(max(0, min(100, soil_score * combined)), 1)

        return jsonify({
            "mode": "suitability_only",
            "note": f"No historical yield data or soil ranges found for '{crop}' — this is NOT a yield prediction, only a soil+rainfall+NDVI suitability indicator.",
            "soil_score": soil_score,
            "soil_grade": soil_grade,
            "rainfall_factor": rainfall_factor,
            "ndvi_factor": ndvi_factor,
            "ndvi_note": ndvi_note,
            "suitability_score": suitability,
        })

    predicted = predict_current_yield(series, soil_score, rainfall_factor, ndvi_factor)

    return jsonify({
        "mode": "yield_estimate",
        "source": source,
        "historical": series,
        "soil_score": soil_score,
        "soil_grade": soil_grade,
        "rainfall_factor": rainfall_factor,
        "ndvi_factor": ndvi_factor,
        "ndvi_note": ndvi_note,
        "current_year": current_year,
        "predicted_yield": predicted,
        "unit": get_unit(crop),
        "disclaimer": "Predicted value is a probable heuristic estimate, not a validated forecast.",
    })


if __name__ == "__main__":
    # use_reloader=False: the watcher was spuriously restarting on stdlib file
    # timestamp changes (Windows quirk), killing in-flight requests. Disabled
    # for stability — restart manually (Ctrl+C, rerun) after editing code.
    app.run(debug=True, port=5000, use_reloader=False)
