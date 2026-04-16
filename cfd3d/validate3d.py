"""3D validation against MacBook Pro 16" thermal-imaging metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .config3d import Config3D
from .energy3d import TemperatureField3D
from .geometry3d import Geometry3D


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
        out = ["=" * 62,
               "  Metric              Sim       Expected      Result",
               "=" * 62]
        for m in self.metrics:
            lo, hi = m.expected
            mark = "PASS" if m.passed else "FAIL"
            out.append(f"  {m.name:18s}  {m.value:6.1f}    [{lo:5.1f}, {hi:5.1f}]   {mark}")
        out.append("=" * 62)
        return out


def compute_stats(config: Config3D, geom: Geometry3D, T: TemperatureField3D) -> Dict[str, float]:
    T_field = T.T

    soc_id = next((i for i, c in enumerate(config.components) if c.name == "SoC"), None)
    if soc_id is not None:
        soc_mask = (geom.comp_id == soc_id)
        soc_mean = float(T_field[soc_mask].mean()) if soc_mask.any() else float("nan")
    else:
        soc_mean = float("nan")

    bat_mask = np.zeros_like(geom.is_solid)
    for i, c in enumerate(config.components):
        if c.name.startswith("Bat"):
            bat_mask |= (geom.comp_id == i)
    battery_max = float(T_field[bat_mask].max()) if bat_mask.any() else float("nan")

    px0, px1 = config.validation.palm_region_x_m
    py0, py1 = config.validation.palm_region_y_m
    # Palm rest = top surface of chassis over the specified x/y footprint.
    top_k = geom.nz - 1
    X2, Y2 = np.meshgrid(geom.x_c, geom.y_c)
    palm_mask_2d = (X2 >= px0) & (X2 <= px1) & (Y2 >= py0) & (Y2 <= py1)
    palm_mean = float(T_field[top_k][palm_mask_2d].mean()) if palm_mask_2d.any() else float("nan")

    # Exhaust temp: rear face cells inside any outlet patch, averaged over z, x.
    exhaust_vals = T_field[:, -1, :][geom.outlet_rear_mask]
    exhaust_mean = float(exhaust_vals.mean()) if exhaust_vals.size else float("nan")

    return {
        "soc_mean_c": soc_mean,
        "battery_max_c": battery_max,
        "palm_mean_c": palm_mean,
        "exhaust_mean_c": exhaust_mean,
    }


def validate(config: Config3D, stats: Dict[str, float]) -> ValidationReport:
    rep = ValidationReport()
    v = config.validation
    rep.metrics.append(MetricResult("SoC mean (C)",       stats["soc_mean_c"],     v.soc_mean_c))
    rep.metrics.append(MetricResult("Battery max (C)",    stats["battery_max_c"],  v.battery_max_c))
    rep.metrics.append(MetricResult("Palm rest mean (C)", stats["palm_mean_c"],    v.palm_mean_c))
    rep.metrics.append(MetricResult("Exhaust mean (C)",   stats["exhaust_mean_c"], v.exhaust_mean_c))
    return rep
