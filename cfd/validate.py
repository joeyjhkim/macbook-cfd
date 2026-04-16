"""Compare simulated thermal metrics to published MacBook Pro 16" data.

Each metric is a single scalar reduced from the temperature field; the
YAML config supplies an acceptable [min, max] range. Returns a structured
report so callers can decide what to do with failures (CI, regression
testing, exit code in run.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .config import Config
from .energy import TemperatureField
from .geometry import Geometry


@dataclass
class MetricResult:
    name: str
    value: float
    expected: Tuple[float, float]

    @property
    def passed(self) -> bool:
        lo, hi = self.expected
        return lo <= self.value <= hi


@dataclass
class ValidationReport:
    metrics: List[MetricResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.metrics)

    def summary_lines(self) -> List[str]:
        out = []
        out.append("=" * 62)
        out.append("  Metric              Sim       Expected      Result")
        out.append("=" * 62)
        for m in self.metrics:
            lo, hi = m.expected
            mark = "PASS" if m.passed else "FAIL"
            out.append(f"  {m.name:18s}  {m.value:6.1f}    [{lo:5.1f}, {hi:5.1f}]   {mark}")
        out.append("=" * 62)
        return out


def compute_stats(config: Config, geom: Geometry, T: TemperatureField) -> Dict[str, float]:
    """Compute scalar metrics used by validation and reporting."""
    T_field = T.T

    # Find SoC component index by name
    soc_id = next(i for i, c in enumerate(config.components) if c.name == "SoC")
    soc_mask = (geom.comp_id == soc_id)
    soc_mean = float(T_field[soc_mask].mean()) if soc_mask.any() else float("nan")

    bat_mask = np.zeros_like(geom.is_solid)
    for i, c in enumerate(config.components):
        if c.name.startswith("Bat"):
            bat_mask |= (geom.comp_id == i)
    battery_max = float(T_field[bat_mask].max()) if bat_mask.any() else float("nan")

    px0, px1 = config.validation.palm_x_m
    py0, py1 = config.validation.palm_y_m
    palm_mask = (geom.X_c >= px0) & (geom.X_c <= px1) & (geom.Y_c >= py0) & (geom.Y_c <= py1)
    palm_mean = float(T_field[palm_mask].mean()) if palm_mask.any() else float("nan")

    # Exhaust temp: cell row j=ny-1 in outlet bands
    exhaust_cells = T_field[-1, geom.outlet_top_mask]
    exhaust_mean = float(exhaust_cells.mean()) if exhaust_cells.size else float("nan")

    return {
        "soc_mean_c": soc_mean,
        "battery_max_c": battery_max,
        "palm_mean_c": palm_mean,
        "exhaust_mean_c": exhaust_mean,
    }


def validate(config: Config, stats: Dict[str, float]) -> ValidationReport:
    rep = ValidationReport()
    v = config.validation
    rep.metrics.append(MetricResult("SoC mean (C)",       stats["soc_mean_c"],     v.soc_mean_c))
    rep.metrics.append(MetricResult("Battery max (C)",    stats["battery_max_c"],  v.battery_max_c))
    rep.metrics.append(MetricResult("Palm rest mean (C)", stats["palm_mean_c"],    v.palm_mean_c))
    rep.metrics.append(MetricResult("Exhaust mean (C)",   stats["exhaust_mean_c"], v.exhaust_mean_c))
    return rep
