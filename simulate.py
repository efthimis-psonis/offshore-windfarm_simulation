from __future__ import annotations

import os
from typing import Callable, Dict, Any, Mapping, Optional, List
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .models import MODEL_SPECS

PowerCurve = Dict[str, Any]

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

def _power_from_spec(ws: float, spec: PowerCurve) -> float:
    # Tabular curve
    if "power_curve" in spec:
        cut_in = float(spec.get("cut_in", 0))
        cut_out = float(spec.get("cut_out", 30))
        if ws < cut_in or ws > cut_out:
            return 0.0
        pc = spec["power_curve"]
        return float(np.interp(ws, pc["wind_speed"], pc["value"]))
    # Parametric
    p_nom = float(spec["p_nom"])
    v_in = float(spec["v_in"])
    v_r = float(spec["v_rated"])
    v_out = float(spec["v_out"])
    if ws < v_in or ws > v_out:
        return 0.0
    if ws < v_r:
        return p_nom * (ws - v_in) / max(v_r - v_in, 1e-9)
    return p_nom

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
    Run a simple power-curve simulation over a time range, aggregating to farm level and (optionally) scoring vs. real production.

    Parameters
    ----------
    real_production_file : CSV with 'Datetime' and one column per wind park (MW)
    ds : xarray Dataset with TIME, U2Z, V2Z (and optionally ROZ)
    positions : (N,2) turbine coordinates (projected meters)
    turbine_ids, parks, turbine_map : parallel mappings (id -> object with .name() -> type_key)
    interpolate_fn : callable(positions, U2Z, V2Z, ...) -> (N,) wind speeds (m/s)
    """
    assert ds is not None, "Provide an xarray Dataset (ds)."
    assert positions is not None and turbine_ids is not None and parks is not None and turbine_map is not None

    # === 1) Real production (optional) ===
    real_df = None
    if real_production_file is not None and os.path.exists(real_production_file):
        real_df = pd.read_csv(real_production_file)
        real_df.columns = [c.strip() for c in real_df.columns]
        if "Datetime" not in real_df.columns:
            raise ValueError("real_production_file must contain a 'Datetime' column")
        real_df["datetime"] = _to_naive_utc(real_df["Datetime"])
        if start_trim:
            real_df = real_df[real_df["datetime"] >= pd.Timestamp(start_trim)].reset_index(drop=True)

    # === 2) Align ds to real_df if present ===
    if real_df is not None and not real_df.empty:
        real_start = np.datetime64(real_df["datetime"].iloc[0])
        real_end = np.datetime64(real_df["datetime"].iloc[-1])
        ds = ds.sel(TIME=slice(real_start, real_end))

    time_len = int(ds.sizes.get("TIME", ds.dims.get("TIME", 0)))
    if real_df is not None and not real_df.empty:
        mini_steps = min(time_len, len(real_df))
    else:
        mini_steps = time_len

    # === 3) Setup output & completion tracking ===
    os.makedirs(output_folder, exist_ok=True)
    completed_models_file = completed_models_file or os.path.join(output_folder, "completed_models.txt")
    completed_models = set()
    if os.path.exists(completed_models_file):
        with open(completed_models_file) as f:
            completed_models = {line.strip() for line in f}

    if real_df is not None and not real_df.empty:
        start_str = real_df["datetime"].iloc[0].strftime("%b%d").lower()
        end_str = real_df["datetime"].iloc[-1].strftime("%b%d").lower()
        model_tag = f"simple_power_curve:{start_str}_to_{end_str}"
    else:
        model_tag = f"simple_power_curve:steps_{mini_steps}"

    if model_tag in completed_models:
        if verbose:
            print(f"⏩ Skipping {model_tag} (already done)")
        return None, None

    # === 4) Iterate and simulate ===
    if "U2Z" not in ds or "V2Z" not in ds:
        raise KeyError("Dataset must contain 'U2Z' and 'V2Z' variables (m/s components at hub or reference height).")

    all_rows = []
    N = len(turbine_ids)
    for t_idx in tqdm(range(mini_steps), desc="Simulating simple power curve", disable=not verbose):
        U2Z = ds["U2Z"].isel(TIME=t_idx).values.ravel()
        V2Z = ds["V2Z"].isel(TIME=t_idx).values.ravel()

        if interpolate_fn is None:
            raise ValueError("Provide interpolate_fn(positions, U2Z, V2Z, ...) that returns (N,) wind speeds.")
        turbine_ws = interpolate_fn(positions, U2Z, V2Z)

        # Use real timestamp if available, else derive from ds.TIME
        if real_df is not None and not real_df.empty:
            ts = real_df["datetime"].iloc[t_idx]
        else:
            ts_val = ds["TIME"].isel(TIME=t_idx).values
            ts = pd.to_datetime(ts_val).to_pydatetime()

        for i in range(N):
            turb_obj = turbine_map[turbine_ids[i]]
            # Accept .name() or str directly
            ttype = turb_obj.name() if hasattr(turb_obj, "name") and callable(turb_obj.name) else str(turb_obj)
            spec = MODEL_SPECS.get(ttype)
            if spec is None:
                raise KeyError(f"MODEL_SPECS missing entry for '{ttype}'")

            ws = float(turbine_ws[i])
            pw = _power_from_spec(ws, spec)

            all_rows.append({
                "datetime": ts,
                "turbine_id": turbine_ids[i],
                "wind_park": parks[i],
                "power_W": pw,
            })

    df_results = pd.DataFrame(all_rows)

    # Minor rename fix (optional)
    df_results["wind_park"] = df_results["wind_park"].replace({
        "Thortntonbank NE Offshore WP": "Thorntonbank NE Offshore WP"
    })

    farm_pivot = (
        df_results.pivot_table(index="datetime", columns="wind_park", values="power_W", aggfunc="sum")
        .reset_index()
    )

    # === 5) Metrics (optional) ===
    metrics_df = None
    if real_df is not None and not real_df.empty:
        # Merge with explicit suffixes to avoid confusion
        park_cols = [c for c in real_df.columns if c not in {"Datetime", "datetime"}]
        comparison = farm_pivot.merge(
            real_df[["datetime"] + park_cols],
            on="datetime",
            how="inner",
            suffixes=("_sim", "_real"),
        )

        recs = []
        for park in farm_pivot.columns[1:]:
            if park not in park_cols:
                if verbose:
                    print(f"⚠️ Skipping park '{park}' — not found in real production data.")
                continue
            sim = comparison[f"{park}_sim"].values  # W
            real = comparison[f"{park}_real"].values * 1e6  # MW -> W
            recs.append({
                "Wind_Park": park,
                "RMSE_MW": _rmse(real, sim) / 1e6,
                "MAE_MW": _mae(real, sim) / 1e6,
                "Bias_MW": float(np.mean(sim - real)) / 1e6,
            })
        metrics_df = pd.DataFrame(recs)

    # === 6) Save outputs ===
    total_minutes = minutes_per_step * max(0, (mini_steps - 1))
    simulation_days = int(np.floor(total_minutes / (24 * 60))) if mini_steps > 0 else 0

    with open(completed_models_file, "a") as f:
        f.write(f"{model_tag}
")

    farm_csv = os.path.join(output_folder, f"farmlevel_{simulation_days}days.csv")
    farm_pivot.to_csv(farm_csv, index=False)

    turb_csv = os.path.join(output_folder, f"turbinelevel_{simulation_days}days.csv")
    df_results.to_csv(turb_csv, index=False)

    if metrics_df is not None:
        metrics_csv = os.path.join(output_folder, f"metrics.csv")
        metrics_df.to_csv(metrics_csv, index=False)

    if verbose:
        print("✅ Finished simulation.")
        print(f"   ├─ farm:    {farm_csv}")
        print(f"   ├─ turbines:{turb_csv}")
        if metrics_df is not None:
            print(f"   └─ metrics: {metrics_csv}")

    return df_results, metrics_df
