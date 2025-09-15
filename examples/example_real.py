"""
Real-data template.
- Builds a KDTree from LAT/LON in the NetCDF (converted to UTM EPSG:32631).
- Interpolates U/V to turbine positions via IDW kNN.
- Maps turbine ids -> model keys present in MODEL_SPECS.
Fill your paths/layout and run:
    python examples/example_real.py
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import xarray as xr

from simulation.models import MODEL_SPECS, normalize_turbine_name
from simulation.interpolate import build_tree_from_dataset, interpolate_with_tree
from simulation.run import run_simulation_for_years

# --- USER: layout CSV with columns: Turbine_ID, Wind_Park, Latitude, Longitude, Turbine_Type
layout_csv = "PATH/TO/full_farm_layout.csv"
nc_template = r"PATH/TO/MAR-ERA5s-{year}.nc"
real_template = r"PATH/TO/real_production_{year}.csv"
output_base = r"outputs/simple_model"

# Load layout and project to UTM
from pyproj import Transformer
df = pd.read_csv(layout_csv)
proj = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
x_utm, y_utm = proj.transform(df["Longitude"].to_numpy(), df["Latitude"].to_numpy())
df["x_utm"] = x_utm
df["y_utm"] = y_utm

positions = df[["x_utm", "y_utm"]].to_numpy()
turbine_ids = df["Turbine_ID"].astype(str).to_list()
parks = df["Wind_Park"].astype(str).to_list()

# Map id -> model key present in MODEL_SPECS (normalize both sides)
def resolve_model_key(raw: str) -> str:
    raw_n = normalize_turbine_name(raw)
    for k in MODEL_SPECS.keys():
        if normalize_turbine_name(k) == raw_n:
            return k
    raise KeyError(f"MODEL_SPECS missing turbine type '{raw}'")
turbine_map = {tid: resolve_model_key(raw) for tid, raw in zip(df["Turbine_ID"], df["Turbine_Type"])}

# Build KDTree once from one year file (assumes same grid across years)
with xr.open_dataset(nc_template.format(year=2022)) as ds0:
    tree, grid_xy = build_tree_from_dataset(ds0)

def my_interp(positions_utm, U2Z_flat, V2Z_flat):
    return interpolate_with_tree(positions_utm, U2Z_flat, V2Z_flat, tree=tree, k=4)

years = [2021, 2022]
results = run_simulation_for_years(
    years=years,
    output_base=output_base,
    nc_path_template=nc_template,
    real_path_template=real_template,
    positions=positions,
    turbine_ids=turbine_ids,
    parks=parks,
    turbine_map=turbine_map,
    interpolate_fn=my_interp,
    minutes_per_step=15,
    start_trim="2021-05-20 00:00:00",
    verbose=True,
)
print("Done. Outputs under:", output_base)
