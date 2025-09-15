# windfarm-simulation

A lean, transparent wind farm simulation package designed for portfolio review and rapid experimentation.  
It demonstrates clean engineering practices (clear structure, type hints, vectorization), simple power-curve modeling, and reproducible results with minimal dependencies.

> This repo includes a **ready-to-run example** and a **drop-in pipeline** that mirrors a realistic workflow using NetCDF weather and farm layouts. It’s tailored to present your applied modeling skills for the ECB traineeship.

## Features
- **Simple power curve** (cut‑in/linear/rated/cut‑out) with optional density scaling.
- **Pluggable interpolation**: bring your own `interpolate_turbine_conditions` for production; a safe placeholder is provided.
- **Deterministic & vectorized**: easy to test and profile.
- **Farm‑level metrics**: RMSE/MAE/Bias vs. real production if provided.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Repository layout

```
windfarm-simulation
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── simulation/
│   ├── __init__.py
│   ├── models.py             # MODEL_SPECS, Turbine, WindPark
│   ├── interpolate.py        # k-NN helpers + placeholder interpolate_turbine_conditions
│   ├── simulate.py           # simulate_simple_power_curve (realistic pipeline-ready)
│   ├── run.py                # run_year_simulation, run_simulation_for_years
│
└── examples/
    ├── example_run.py        # runnable synthetic example (no data needed)
    └── example_real.py       # skeleton showing how to wire real paths & maps
```

## Quick start (synthetic demo)

```bash
python examples/example_run.py
```
This writes a CSV under `outputs/` with a small 3‑turbine demo.

## Using real data

1. Fill `MODEL_SPECS` in `simulation/models.py` with real turbine types. Two formats are supported:
   - **Parametric** (`p_nom`, `v_in`, `v_rated`, `v_out`), or
   - **Tabular power curve** (`power_curve`: arrays of `wind_speed` and `value` in Watts, plus `cut_in`, `cut_out`).

2. Implement or import your **interpolation**:
   - Plug your function into `simulate_simple_power_curve(..., interpolate_fn=your_fn)`.
   - The default placeholder returns a spatial mean wind speed (for smoke tests only).

3. Prepare inputs:
   - **NetCDF** with `TIME` and wind components `U2Z`, `V2Z` (and optionally `ROZ` for density).
   - **Real production CSV** with a `Datetime` column and a column per wind park (in **MW**).

4. Use `examples/example_real.py` as a template.

## Notes & assumptions
- Times are assumed **naive UTC** after conversion. Adjust if your CSV uses local time.
- Metrics are computed after aligning `datetime` across sim/real. Missing parks are skipped.
- To avoid scikit‑learn version drift, RMSE/MAE are computed with NumPy/Pandas.

## License
MIT
