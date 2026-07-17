# Yield Estimator — Simple Dashboard

## Run it

```
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## Data priority (as specified)

1. District-level (`data/apy_district.csv`, parsed from your DES xls, 2020-21 to 2022-23,
   34 crops — mostly horticulture/commercial crops) is tried first.
2. If no matching non-null rows for that exact State+District+Crop, falls back to
   state-level average from `data/crop_yield.csv` (1997-2020, 55 crops).

**Known gap**: the xls file does not include Rice, and has effectively no data for Wheat
in major wheat states. Any query for those crops will resolve to the state-level CSV
fallback, not district-level. This is expected given the fallback rule — flag it in your
demo rather than let a reviewer discover it.

## Live NDVI setup (optional — degrades gracefully if skipped)

The app tries to pull live NDVI via Google Earth Engine. Without credentials configured,
it silently falls back to a neutral factor (no crash) — you'll see `"ndvi_factor": null`
in the response. To make it live:

1. Sign up for a free GEE account: https://code.earthengine.google.com/register
2. Create a GCP service account, download its JSON key, enable the Earth Engine API for
   that project.
3. Before running the app: `earthengine authenticate` (or set up the service-account
   credentials per the `earthengine-api` docs) so `ee.Initialize()` succeeds.

Given this needs its own GCP setup, budget real time for it if you want it live for
tomorrow's demo — don't leave it to the last hour.

## What "predicted current yield" actually is

It's `historical_average * weighted_adjustment`, where the adjustment blends:
- soil suitability score (from the agronomist's formula) vs. a "Good"-grade neutral point
- live rainfall this year vs. historical average rainfall for that state+crop
- live NDVI vs. a generic reference value (0.5) — **not** a real per-district historical
  NDVI baseline, since that data isn't wired up yet. Replace this with an actual
  historical NDVI average per district once you have one (you already have the pull
  script from earlier — join it in here for a real comparison instead of the placeholder).

This is a transparent heuristic, not a trained/validated model. State that plainly if
asked how accurate it is — "probable estimate" was the agreed scope, and that's what
this delivers.

## Regenerating district data from a new xls export

```
python data_prep.py path/to/new_report.xls
```
