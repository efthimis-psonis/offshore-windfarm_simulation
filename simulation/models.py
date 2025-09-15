from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any

def format_power_curve(wind_speeds, power_values):
    """Return a DataFrame with power in W (inputs may be MW)."""
    return pd.DataFrame({"wind_speed": wind_speeds, "value": [p * 1000 if p < 1e5 else p for p in power_values]})

def normalize_turbine_name(name: str) -> str:
    return str(name).lower().replace(" ", "").replace("-", "")

# --- MODEL_SPECS (tabular) ----------------------------------------------------
# Keys are the *display names*. You may look them up by normalize_turbine_name(key).
MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "Vestas V90-3": {
        "nominal_power": 3e6,
        "hub_height": (65+105)/2,
        "rotor_diameter": 90,
        "cut_in": 3.5,
        "cut_out": 25,
        "power_curve": format_power_curve(
            [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 25],
            [38, 77, 133, 190, 271, 353, 467, 581, 733, 886, 1079, 1272, 1484, 1696, 1901, 2106, 2298, 2489, 2643, 2797, 2874, 2951, 2972, 2993, 2996, 2999, 3000, 3000]
        )
    },
    "Siemens SG 8-167DD": {
        "nominal_power": 8e6,
        "hub_height": 92,
        "rotor_diameter": 167,
        "cut_in": 3.5,
        "cut_out": 25,
        "power_curve": format_power_curve(
            [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 15, 16, 17, 18, 19, 20, 25],
            [p * (8.711111111/8) for p in [48, 169, 350, 593, 930, 1307, 1737, 2186, 2730, 3278, 3980, 4687, 5400, 6112, 6690, 7249, 7570, 7795, 7895, 7947, 7990, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000]]
        )
    },
    "Siemens Gamesa SG 8.0-167 DD": {
        "nominal_power": 8e6,
        "hub_height": 92,
        "rotor_diameter": 167,
        "cut_in": 3.5,
        "cut_out": 25,
        "power_curve": format_power_curve(
            [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 15, 16, 17, 18, 19, 20, 25],
            [48, 169, 350, 593, 930, 1307, 1737, 2186, 2730, 3278, 3980, 4687, 5400, 6112, 6690, 7249, 7570, 7795, 7895, 7947, 7990, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000]
        )
    },
    "Vestas V112-3.3": {
        "nominal_power": 3.3e6,
        "hub_height": (84+119)/2,
        "rotor_diameter": 112,
        "cut_in": 3,
        "cut_out": 25,
        "power_curve": format_power_curve(
            [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 25],
            [22.0, 73.0, 134.0, 209.0, 302.0, 415.0, 552.0, 714.0, 906.0, 1123.0, 1370.0, 1648.0, 1950.0, 2268.0, 2586.0, 2868.0, 3071.0, 3201.0, 3266.0, 3291.0, 3298.0, 3299.0, 3300.0, 3300.0]
        )
    },
    "Vestas V112-3": {
        "nominal_power": 3e6,
        "hub_height": (84+119)/2,
        "rotor_diameter": 112,
        "cut_in": 2.5,
        "cut_out": 25,
        "power_curve": format_power_curve(
            [2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 25],
            [10, 19, 71, 122, 212, 303, 428, 554, 753, 953, 1153, 1375, 1648, 1944, 2221, 2498, 2717, 2936, 2985, 3000, 3000]
        )
    },
    "Vestas V164-8400": {
        "nominal_power": 8e6,
        "hub_height": (105+140)/2,
        "rotor_diameter": 164,
        "cut_in": 3.5,
        "cut_out": 25,
        "power_curve": format_power_curve(
            [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 25],
            [p * (8.4/8) for p in [40, 100, 370, 650, 895, 1150, 1500, 1850, 2375, 2900, 3525, 4150, 4875, 5600, 6350, 7100, 7580, 7800, 7920, 8000, 8000]]
        )
    },
    "Vestas V164-9.5": {
        "nominal_power": 9.5e6,
        "hub_height": (105+140)/2,
        "rotor_diameter": 164,
        "cut_in": 3.5,
        "cut_out": 25,
        "power_curve": format_power_curve(
            [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 25],
            [115, 249, 430, 613, 900, 1226, 1600, 2030, 2570, 3123, 3784, 4444, 5170, 5900, 6600, 7299, 7960, 8601, 9080, 9272, 9410, 9500, 9500]
        )
    },
    "Siemens D7": {
        "nominal_power": 7e6,
        "hub_height": 140,
        "rotor_diameter": 154,
        "cut_in": 3.5,
        "cut_out": 30,
        "power_curve": format_power_curve(
            [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 30],
            [(p * 7.166667/8) for p in [48, 169, 350, 593, 930, 1307, 1737, 2186, 2730, 3278, 3980, 4687, 5400, 6112, 6690, 7249, 7570, 7795, 7895, 7947, 7990, 8000, 8000]]
        )
    },
    "Senvion 126": {
        "nominal_power": 6.15e6,
        "hub_height": (85+117)/2,
        "rotor_diameter": 126,
        "cut_in": 3.5,
        "cut_out": 30,
        "power_curve": format_power_curve(
            [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 30],
            [p * (6.15/8) for p in [48, 169, 350, 593, 930, 1307, 1737, 2186, 2730, 3278, 3980, 4687, 5400, 6112, 6690,7249, 7570, 7795, 7895, 7947, 7990, 8000, 8000]]
        )
    },
    "Repower": {
        "nominal_power": 5e6,
        "hub_height": (90+120)/2,
        "rotor_diameter": 126,
        "cut_in": 3.5,
        "cut_out": 30,
        "power_curve": format_power_curve(
            [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 30],
            [70.0, 141.0, 240.0, 343.0, 490.0, 636.0, 850.0, 1067.0, 1340.0, 1615.0, 1950.0, 2289.0, 2720.0, 3166.0, 3575.0, 3984.0, 4366.0, 4748.0, 4930.0, 4978.0, 4990.0, 4999.0, 5000.0, 5000.0]
        )
    },
    "Haliade 150": {
        "nominal_power": 3e6,
        "hub_height": 100,
        "rotor_diameter": 151,
        "cut_in": 3.5,
        "cut_out": 30,
        "power_curve": format_power_curve(
            [3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 25],
            [100, 220, 320, 440, 575, 721, 945, 1173, 1485, 1796, 2157, 2517, 2940, 3360, 3930, 4485, 5160, 5792, 5960, 6000, 6000]
        )
    },
}

# Optional helpers to build PyWake objects (only if py_wake installed)
def build_pywake_turbine_models():
    try:
        from py_wake.wind_turbines import WindTurbine, WindTurbines
        from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
    except Exception:
        raise ImportError("py_wake not installed; install it or skip this helper.")

    models = {}
    for model_name, specs in MODEL_SPECS.items():
        pc = specs["power_curve"]
        ws = pc["wind_speed"].values if hasattr(pc["wind_speed"], "values") else np.asarray(pc["wind_speed"])
        pw = pc["value"].values if hasattr(pc["value"], "values") else np.asarray(pc["value"])
        models[normalize_turbine_name(model_name)] = WindTurbine(
            name=model_name,
            diameter=specs["rotor_diameter"],
            hub_height=specs["hub_height"],
            powerCtFunction=PowerCtTabular(ws=ws, power=pw, power_unit="w", ct=np.ones_like(ws) * 0.8),
        )
    wt = list(models.values())
    return models, WindTurbines(
        names=[t.name for t in wt],
        diameters=[t.diameter() for t in wt],
        hub_heights=[t.hub_height() for t in wt],
        powerCtFunctions=[t.powerCtFunction for t in wt],
    )
