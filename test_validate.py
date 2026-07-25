import urllib.request
import json

tests = [
    ("Bagalkot", "Brinjal",      {"soc": 0.5, "pH": 7.2, "N": 180, "P": 12, "K": 130}),
    ("Ludhiana", "Wheat",        {"soc": 0.6, "pH": 7.0, "N": 250, "P": 25, "K": 180}),
    ("Idukki",   "Cardamom",     {"soc": 1.5, "pH": 5.5, "N": 350, "P": 30, "K": 200}),
    ("Guntur",   "Chilli",       {"soc": 0.4, "pH": 7.5, "N": 200, "P": 20, "K": 150}),
    ("Pune",     "Sugarcane",    {"soc": 0.8, "pH": 6.8, "N": 280, "P": 35, "K": 200}),
    ("Jhunjhunu","Rapeseed",     {"soc": 0.3, "pH": 8.0, "N": 120, "P": 10, "K": 100}),
    ("West Godavari", "Rice",    {"soc": 0.6, "pH": 6.5, "N": 260, "P": 28, "K": 170}),
    ("Alwar",    "Wheat",        {"soc": 0.3, "pH": 7.8, "N": 150, "P": 15, "K": 120}),
]

for dist, crop, soil in tests:
    params = f"district={dist}&crop={crop}&soc={soil['soc']}&pH={soil['pH']}&N={soil['N']}&P={soil['P']}&K={soil['K']}"
    url = f"http://127.0.0.1:8000/api/predict/?{params}"
    try:
        r = urllib.request.urlopen(url, timeout=10)
        d = json.loads(r.read())
        if d.get("status") == "available":
            x = d["data"]
            print(f"{dist:18s} {crop:14s} -> {x['predicted_yield']:>8.0f} kg/ha  (pot={x['potential_yield']:.0f} soil={x['soil_factor']:.2f} water={x['water_factor']:.2f} ndvi={x['ndvi_factor']:.2f})")
        else:
            print(f"{dist:18s} {crop:14s} -> {d.get('message','?')}")
    except Exception as e:
        print(f"{dist:18s} {crop:14s} -> ERR: {e}")
