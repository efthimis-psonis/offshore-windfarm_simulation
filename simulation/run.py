from __future__ import annotations

import os
from typing import Callable, Dict, Any, Mapping, Iterable, Optional, List
import xarray as xr
import numpy as np

from .simulate import simulate_simple_power_curve

def run_year_simulation(
    year: int,
    nc_path: str,
    real_path: Optional[str],
    output_base: str,
    positions: np.ndarray,
    turbine_ids: List[str],
    parks: List[str],
    turbine_map: Mapping[str, Any],
    interpolate_fn: Callable[..., np.ndarray],
    minutes_per_step: int = 15,
    start_trim: str = "2021-05-20 00:00:00",
    verbose: bool = True,
):
    if verbose:
        print(f"\n📅 Running simulation for year {year}...")
    ds = xr.open_dataset(nc_path)

    out_dir = os.path.join(output_base, f"simple_model_{year}")
    os.makedirs(out_dir, exist_ok=True)

    rp = real_path if (real_path and os.path.exists(real_path)) else None
    if real_path and not os.path.exists(real_path) and verbose:
        print(f"⚠️ Real production not found for {year}; running without metrics.")

    return simulate_simple_power_curve(
        output_folder=out_dir,
        real_production_file=rp,
        ds=ds,
        positions=positions,
        turbine_ids=turbine_ids,
        parks=parks,
        turbine_map=turbine_map,
        minutes_per_step=minutes_per_step,
        interpolate_fn=interpolate_fn,
        sanitize_fn=lambda s: s.replace(" ", "_") if isinstance(s, str) else s,
        completed_models_file=os.path.join(out_dir, "completed_models.txt"),
        start_trim=start_trim,
        verbose=verbose,
    )

def run_simulation_for_years(
    years: Iterable[int],
    output_base: str,
    nc_path_template: str,
    real_path_template: Optional[str],
    positions: np.ndarray,
    turbine_ids: List[str],
    parks: List[str],
    turbine_map: Mapping[str, Any],
    interpolate_fn: Callable[..., np.ndarray],
    minutes_per_step: int = 15,
    start_trim: str = "2021-05-20 00:00:00",
    verbose: bool = True,
):
    results = {}
    for year in years:
        nc_path = nc_path_template.format(year=year)
        rp = real_path_template.format(year=year) if real_path_template else None
        try:
            results[year] = run_year_simulation(
                year=year,
                nc_path=nc_path,
                real_path=rp,
                output_base=output_base,
                positions=positions,
                turbine_ids=turbine_ids,
                parks=parks,
                turbine_map=turbine_map,
                interpolate_fn=interpolate_fn,
                minutes_per_step=minutes_per_step,
                start_trim=start_trim,
                verbose=verbose,
            )
        except Exception as e:
            print(f"❌ Simulation failed for {year}: {e}")
            results[year] = (None, None)
    return results
