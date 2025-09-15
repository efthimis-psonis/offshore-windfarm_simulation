from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

# --- Replace/extend with your real models ------------------------------------
# Supports both parametric and tabular power curves.
# Parametric: use keys p_nom, v_in, v_rated, v_out
# Tabular:   provide "power_curve": {"wind_speed": [...], "value": [...]}, and cut_in/cut_out
MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "example_parametric": {
        "p_nom": 8.4e6,   # W
        "v_in": 3.5,      # m/s
        "v_rated": 12.0,  # m/s
        "v_out": 25.0,    # m/s
    },
    "example_tabular": {
        "cut_in": 3.5,
        "cut_out": 25.0,
        "power_curve": {
            # Minimal illustrative curve (m/s -> W). Replace with real arrays.
            "wind_speed": np.array([0, 3.5, 7, 12, 25], dtype=float),
            "value":      np.array([0,    0, 3e6, 8.4e6, 0], dtype=float),
        }
    },
}

@dataclass(frozen=True)
class Turbine:
    """Minimal turbine representation. Use your mapping logic externally if needed."""
    id: str
    x: float  # meters (projected)
    y: float  # meters (projected)
    type_key: str

@dataclass
class WindPark:
    name: str
    turbines: List[Turbine]

    def positions(self) -> np.ndarray:
        return np.array([[t.x, t.y] for t in self.turbines], dtype=float)

    def type_keys(self) -> List[str]:
        return [t.type_key for t in self.turbines]

    def ids(self) -> List[str]:
        return [t.id for t in self.turbines]
