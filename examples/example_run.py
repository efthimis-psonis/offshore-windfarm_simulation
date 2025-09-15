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
LAT = = np.array([
    [51.452194, 51.450554, 51.44887,  51.447147, 51.445374, 51.443565],
    [51.497063, 51.49542,  51.49374,  51.492012, 51.49023,  51.488422],
    [51.54193,  51.540295, 51.5386,   51.53687,  51.535095, 51.533276],
    [51.5868,   51.58516,  51.583466, 51.58173,  51.579952, 51.578133],
    [51.63167,  51.63002,  51.628326, 51.6266,   51.624813, 51.622993],
    [51.676537, 51.674885, 51.673187, 51.671455, 51.66967,  51.667847],
    [51.721405, 51.719753, 51.718052, 51.716316, 51.71453,  51.7127],
    [51.766266, 51.76461,  51.76292,  51.761177, 51.75939,  51.757557]
])    
LON = np.array([
    [2.666789,  2.738783,  2.8107777, 2.8827698, 2.9547434, 3.02672],
    [2.669381,  2.7414527, 2.813516,  2.8855748, 2.957623,  3.0296712],
    [2.6719875, 2.7441318, 2.8162591, 2.8883893, 2.9605079, 3.032626],
    [2.6745994, 2.7468112, 2.8190134, 2.8912103, 2.9634042, 3.0355875],
    [2.6772141, 2.7494948, 2.8217754, 2.8940427, 2.966298,  3.0385523],
    [2.679837,  2.7521884, 2.8245327, 2.8968728, 2.9692028, 3.0415323],
    [2.6824718, 2.754887,  2.827306,  2.8997138, 2.9721208, 3.044512],
    [2.685105,  2.7575943, 2.8300812, 2.9025583, 2.9750373, 3.0475028]
])  
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
