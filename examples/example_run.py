"""
Synthetic demo: no external data required.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import xarray as xr

from simulation.models import MODEL_SPECS
from simulation.simulate import simulate_simple_power_curve

# Positions: 3 turbines in a triangle (UTM meters, arbitrary demo coords)
positions = np.array([[0.0, 0.0], [500.0, 0.0], [0.0, 500.0]])
turbine_ids = ["T1", "T2", "T3"]
parks = ["DemoPark", "DemoPark", "DemoPark"]

# Map ids to spec keys (here: all Vestas V112-3.3)
turbine_map = {tid: "Vestas V112-3.3" for tid in turbine_ids}

# Fake NetCDF with TIME, U2Z, V2Z, LAT/LON (LAT/LON not used by this demo interpolation)
T = 24 * 7
time = pd.date_range("2022-01-01", periods=T, freq="H")
rng = np.random.default_rng(42)
U = rng.normal(0, 1, size=(T, 1)) + 6.0
V = rng.normal(0, 1, size=(T, 1))
LAT = np.array([[51.0]])
LON = np.array([[3.0]])
ds = xr.Dataset(
    data_vars=dict(U2Z=(("TIME", "G"), U), V2Z=(("TIME", "G"), V), LAT=(("Y","X"), LAT), LON=(("Y","X"), LON)),
    coords=dict(TIME=time, G=np.arange(1), Y=np.arange(1), X=np.arange(1)),
)

# Simple interpolation: use the single grid-point speed for all turbines
def demo_interp(positions, U2Z_flat, V2Z_flat):
    ws = np.sqrt(U2Z_flat**2 + V2Z_flat**2).mean()
    return np.full(len(positions), float(ws))

out_dir = "outputs/demo"
os.makedirs(out_dir, exist_ok=True)

df_results, metrics_df = simulate_simple_power_curve(
    output_folder=out_dir,
    real_production_file=None,
    ds=ds,
    positions=positions,
    turbine_ids=turbine_ids,
    parks=parks,
    turbine_map=turbine_map,
    minutes_per_step=60,
    interpolate_fn=demo_interp,
    verbose=True,
)

print(df_results.head())
print("Wrote CSVs to:", out_dir)
