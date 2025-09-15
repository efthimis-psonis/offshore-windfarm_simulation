from __future__ import annotations
import os
import pandas as pd
from pyproj import Transformer
from simulation.models import normalize_turbine_name
from data.belgium.wind_park_data import WIND_PARK_DATA

OUT_DIR = "data/derived"
OUT_CSV = os.path.join(OUT_DIR, "full_farm_layout.csv")

def main():
    rows = []
    for park_name, park in WIND_PARK_DATA.items():
        tids = list(park["Turbine_ID"])
        lats = list(park["Latitude"])
        lons = list(park["Longitude"])
        ttype = park["turbine_type"]
        n = min(len(tids), len(lats), len(lons), len(ttype) if isinstance(ttype, list) else 10**9)
        if not (len(tids) == len(lats) == len(lons) == (len(ttype) if isinstance(ttype, list) else len(tids))):
            print(f"⚠️  Length mismatch in '{park_name}': "
                  f"TIDs={len(tids)}, lat={len(lats)}, lon={len(lons)}, ttype={'list('+str(len(ttype))+')' if isinstance(ttype, list) else 'scalar'} -> using first {n}")
        for i in range(n):
            raw_type = ttype[i] if isinstance(ttype, list) else ttype
            rows.append({
                "Turbine_ID": str(tids[i]),
                "Wind_Park": str(park_name),
                "Latitude": float(lats[i]),
                "Longitude": float(lons[i]),
                "Turbine_Type": str(raw_type),
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["Turbine_ID"]).reset_index(drop=True)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    x_utm, y_utm = transformer.transform(df["Longitude"].to_numpy(), df["Latitude"].to_numpy())
    df["x_utm"] = x_utm
    df["y_utm"] = y_utm
    df["Normalized_Type"] = df["Turbine_Type"].map(normalize_turbine_name)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote layout CSV → {OUT_CSV}  (rows: {len(df)})")

if __name__ == "__main__":
    main()
