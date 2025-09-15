from __future__ import annotations

import os
from typing import Callable, Dict, Any, Mapping, Optional, List
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .models import MODEL_SPECS

def _to_naive_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_convert(None)

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.mean(np.abs(a - b)))

def _power_from_tabular(ws: float, spec: Dict[str, Any]) -> float:
    """Handle DataFrame or dict-like 'power_curve' structures + cut-in/out."""
    cut_in = float(spec.get("cut_in", 0))
    cut_out = float(spec.get("cut_out", 30))
    if ws < cut_in or ws > cut_out:
        return 0.0
    pc = spec["power_curve"]
    if isinstance(pc, pd.DataFrame):
        ws_arr = pc["wind_speed"].to_numpy()
        pw_arr = pc["value"].to_numpy()
    else:
        ws_arr = np.asarray(pc["wind_speed"])
        pw_arr = np.asarray(pc["value"])
    return float(np.interp(ws, ws_arr, pw_arr))

REQUIRED_REAL_COLS = {"Datetime"}  # plus park columns if you expect them

def _validate_real_df(df: pd.DataFrame):
    missing = REQUIRED_REAL_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Real production CSV missing columns: {sorted(missing)}. "
                         f"Got: {sorted(df.columns)}")

def simulate_simple_power_curve(
    output_folder: str,
    real_production_file: Optional[str] = None,
    ds: Optional[xr.Dataset] = None,
    positions: Optional[np.ndarray] = None,
    turbine_ids: Optional[List[str]] = None,
    parks: Optional[List[str]] = None,
    turbine_map: Optional[Mapping[str, Any]] = None,
    minutes_per_step: int = 15,
    interpolate_fn: Optional[Callable[..., np.ndarray]] = None,
    sanitize_fn: Optional[Callable[[str], str]] = None,
    completed_models_file: Optional[str] = None,
    start_trim: str = "2021-05-20 00:00:00",
    verbose: bool = True,
):
    """
    Simulate per-turbine power from NetCDF wind components using tabular power curves.

    - real_production_file: CSV with 'Datetime' and park columns in MW (optional)
    - ds: xarray Dataset with TIME, U2Z, V2Z, and LAT/LON for grid (LAT/LON only needed by your interpolate_fn)
    - positions: (N,2) turbine UTM coordinates
    - turbine_map: id -> object (with .name() or str key) used to look up MODEL_SPECS
    - interpolate_fn: callable(positions, U2Z_flat, V2Z_flat, ...) -> (N,) wind speeds
    """
    assert ds is not None, "Provide an xarray Dataset (ds)."
    assert positions is not None and turbine_ids is not None and parks is not None and turbine_map is not None

    # 1) Real production (optional)
    real_df = None
    if real_production_file is not None and os.path.exists(real_production_file):
        real_df = pd.read_csv(real_production_file)
        real_df.columns = [c.strip() for c in real_df.columns]
        if "Datetime" not in real_df.columns:
            raise ValueError("real_production_file must contain a 'Datetime' column")
        real_df["datetime"] = _to_naive_utc(real_df["Datetime"])
        if start_trim:
            real_df = real_df[real_df["datetime"] >= pd.Timestamp(start_trim)].reset_index(drop=True)

    # 2) Align ds to real_df if present
    if real_df is not None and not real_df.empty:
        real_start = np.datetime64(real_df["datetime"].iloc[0])
        real_end = np.datetime64(real_df["datetime"].iloc[-1])
        ds = ds.sel(TIME=slice(real_start, real_end))

    time_len = int(ds.sizes.get("TIME", ds.dims.get("TIME", 0)))
    if real_df is not None and not real_df.empty:
        mini_steps = min(time_len, len(real_df))
    else:
        mini_steps = time_len

    # 3) Output setup and "already done" guard
    os.makedirs(output_folder, exist_ok=True)
    completed_models_file = completed_models_file or os.path.join(output_folder, "completed_models.txt")
    done = set()
    if os.path.exists(completed_models_file):
        with open(completed_models_file) as f:
            done = {line.strip() for line in f}

    if real_df is not None and not real_df.empty:
        tag = f"{real_df['datetime'].iloc[0].strftime('%b%d').lower()}_{real_df['datetime'].iloc[-1].strftime('%b%d').lower()}"
    else:
        tag = f"steps_{mini_steps}"
    model_tag = f"simple_power_curve:{tag}"
    if model_tag in done:
        if verbose:
            print(f"⏩ Skipping {model_tag} (already done)")
        return None, None

    # 4) Iterate timesteps
    if "U2Z" not in ds or "V2Z" not in ds:
        raise KeyError("Dataset must contain 'U2Z' and 'V2Z' variables.")

    rows = []
    N = len(turbine_ids)

    for t_idx in tqdm(range(mini_steps), desc="Simulating power (tabular curves)", disable=not verbose):
        U2Z = ds["U2Z"].isel(TIME=t_idx).values.ravel()
        V2Z = ds["V2Z"].isel(TIME=t_idx).values.ravel()

        if interpolate_fn is None:
            raise ValueError("Provide interpolate_fn(positions, U2Z_flat, V2Z_flat, ...) -> (N,) wind speeds")

        # Support interpolate functions returning ws or (ws, ...)
        out = interpolate_fn(positions, U2Z, V2Z)
        ws = out[0] if isinstance(out, (tuple, list)) else out
        ws = np.asarray(ws, float)

        ts = (
            real_df["datetime"].iloc[t_idx]
            if (real_df is not None and not real_df.empty)
            else pd.to_datetime(ds["TIME"].isel(TIME=t_idx).values).to_pydatetime()
        )

        for i in range(N):
            t_obj = turbine_map[turbine_ids[i]]
            key = t_obj.name() if hasattr(t_obj, "name") and callable(t_obj.name) else str(t_obj)
            spec = MODEL_SPECS.get(key)
            if spec is None:
                # try normalized
                # map normalized keys -> canonical names
                from .models import normalize_turbine_name, MODEL_SPECS as SPECS
                norm = normalize_turbine_name(key)
                hit = None
                for k in SPECS.keys():
                    if normalize_turbine_name(k) == norm:
                        hit = k
                        break
                if hit is None:
                    raise KeyError(f"MODEL_SPECS missing entry for '{key}'")
                spec = SPECS[hit]

            pw = _power_from_tabular(float(ws[i]), spec)
            rows.append({"datetime": ts, "turbine_id": turbine_ids[i], "wind_park": parks[i], "power_W": pw})

    df_results = pd.DataFrame(rows)

    # Fix common name typo
    df_results["wind_park"] = df_results["wind_park"].replace({"Thortntonbank NE Offshore WP": "Thorntonbank NE Offshore WP"})

    farm = df_results.pivot_table(index="datetime", columns="wind_park", values="power_W", aggfunc="sum").reset_index()

    # 5) Metrics
    metrics_df = None
    if real_df is not None and not real_df.empty:
        park_cols = [c for c in real_df.columns if c not in {"Datetime", "datetime"}]
        cmp = farm.merge(real_df[["datetime"] + park_cols], on="datetime", how="inner", suffixes=("_sim", "_real"))
        recs = []
        for park in farm.columns[1:]:
            if park not in park_cols:
                if verbose:
                    print(f"⚠️ Skipping park '{park}' — not in real production CSV.")
                continue
            sim = cmp[f"{park}_sim"].to_numpy()
            real = cmp[f"{park}_real"].to_numpy() * 1e6  # MW -> W
            recs.append({
                "Wind_Park": park,
                "RMSE_MW": _rmse(real, sim) / 1e6,
                "MAE_MW": _mae(real, sim) / 1e6,
                "Bias_MW": float(np.mean(sim - real)) / 1e6,
            })
        metrics_df = pd.DataFrame(recs)

    # 6) Save
    total_minutes = minutes_per_step * max(0, (mini_steps - 1))
    sim_days = int(np.floor(total_minutes / (24 * 60))) if mini_steps > 0 else 0

    with open(completed_models_file, "a") as f:
        f.write(f"{model_tag}\n")

    farm_csv = os.path.join(output_folder, f"farmlevel_{sim_days}days.csv")
    turb_csv = os.path.join(output_folder, f"turbinelevel_{sim_days}days.csv")
    df_results.to_csv(turb_csv, index=False)
    farm.to_csv(farm_csv, index=False)

    if metrics_df is not None:
        metrics_csv = os.path.join(output_folder, "metrics.csv")
        metrics_df.to_csv(metrics_csv, index=False)

    if verbose:
        print("✅ Finished simulation.")
        print(f"   ├─ farm:    {farm_csv}")
        print(f"   ├─ turbines:{turb_csv}")
        if metrics_df is not None:
            print(f"   └─ metrics: {metrics_csv}")

    return df_results, metrics_df
