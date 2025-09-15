# windfarm-simulation-ecb

A lean, transparent wind farm simulator showcasing:
- **Tabular power curves** for real turbine models (MW→W converted).
- **KDTree+UTM interpolation** from gridded NetCDF (LAT/LON) to turbine positions.
- **Per‑turbine → farm aggregation** with clean metrics vs real production (optional).
- **Deterministic & vectorized** pipeline, minimal dependencies.

> Built to serve as a portfolio artifact for an ECB traineeship application.

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Structure
```
windfarm-simulation-ecb
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── simulation/
│   ├── __init__.py
│   ├── models.py          # MODEL_SPECS (tabular curves), normalize_turbine_name, (optional) PyWake helpers
│   ├── interpolate.py     # build KDTree from NetCDF (LAT/LON→UTM) + IDW kNN interpolation
│   ├── simulate.py        # simulate_simple_power_curve (robust, aligns real vs ds, saves CSVs, metrics)
│   ├── run.py             # run_year_simulation, run_simulation_for_years
│
└── examples/
    ├── example_run.py     # synthetic, runs anywhere
    └── example_real.py    # template wiring real NetCDF + layout + interpolation + mapping
```

## Quick demo (synthetic, no data)
```bash
python examples/example_run.py
```
Outputs are written under `outputs/demo/`.

## Real data template
Edit paths in `examples/example_real.py`: point to your NetCDF template and layout CSV.
Implement or keep `my_interp` (KDTree-based) and ensure your turbine type keys match `MODEL_SPECS`.

## Notes
- Real production CSV must have a `Datetime` column and park columns in **MW**.
- We do **not** depend on scikit‑learn for metrics (pure NumPy RMSE/MAE) to avoid version drift.
- `py_wake` is optional; if installed, helper functions can build `WindTurbine` / `WindTurbines` from specs.

## License
MIT


## Data & tools (Belgian offshore example)

Build the canonical layout CSV from the curated dictionary:
```bash
python tools/build_layout_from_dict.py
# writes data/derived/full_farm_layout.csv
```

Then run the real example (after setting NetCDF and production paths inside the file if needed):
```bash
python examples/example_real.py
```

