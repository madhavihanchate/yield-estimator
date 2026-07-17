"""
Re-parses a DES 'horizontal_crop_vertical_year' .xls export into a clean long-format CSV.
Run this whenever you download a new/updated report from the DES portal:

    python data_prep.py path/to/new_report.xls

Handles the rowspan-based State/District/Year cells, and reads crop names + units
directly from the real <th> header cells (verified against MODIS/known regional data
before trusting this order — see conversation notes).
"""
import sys
import re
from bs4 import BeautifulSoup
import pandas as pd


def parse_apy_xls(path):
    with open(path, "r", errors="ignore") as f:
        content = f.read()
    soup = BeautifulSoup(content, "lxml")
    table = soup.find("table")

    crop_names = re.findall(r"\[crop_name\] => (.+)", str(table))
    session_names = re.findall(r"\[session_name\] => (.+)", str(table))
    crop_session = list(zip(crop_names, session_names))
    n_crops = len(crop_session)

    tbody = table.find("tbody")
    rows = tbody.find_all("tr")

    n_leading = 3  # State, District, Year
    n_cols = n_leading + n_crops * 3

    pending = {}
    records = []
    for r in rows:
        tds = r.find_all("td")
        td_iter = iter(tds)
        row_vals = [None] * n_cols
        col = 0
        while col < n_cols:
            if col in pending and pending[col][0] > 0:
                rem, val = pending[col]
                row_vals[col] = val
                pending[col] = (rem - 1, val)
                if pending[col][0] == 0:
                    del pending[col]
                col += 1
                continue
            try:
                td = next(td_iter)
            except StopIteration:
                break
            val = td.get_text(strip=True)
            rowspan = int(td.get("rowspan", 1))
            row_vals[col] = val
            if rowspan > 1:
                pending[col] = (rowspan - 1, val)
            col += 1

        state, district, year = row_vals[0], row_vals[1], row_vals[2]
        if not year:
            continue
        for i, (crop, session) in enumerate(crop_session):
            base = n_leading + i * 3
            area, prod, yld = row_vals[base], row_vals[base + 1], row_vals[base + 2]
            if area or prod or yld:
                records.append({
                    "state": state, "district": district, "year": year,
                    "crop": crop, "session": session,
                    "area": area, "production": prod, "yield": yld,
                })
    return pd.DataFrame(records)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python data_prep.py path/to/report.xls")
        sys.exit(1)
    df = parse_apy_xls(sys.argv[1])
    out_path = "data/apy_district.csv"
    df.to_csv(out_path, index=False)
    print(f"Parsed {len(df)} rows -> {out_path}")
