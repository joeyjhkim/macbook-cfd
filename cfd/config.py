"""Configuration loading and validation for the CFD case.

A `Config` is an immutable view of the YAML file. Construction validates
geometry (no overlapping components, all boxes inside the domain) so the
solver can trust its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import yaml


@dataclass(frozen=True)
class Component:
    name: str
    x_range_m: Tuple[float, float]
    y_range_m: Tuple[float, float]
    q_watts: float
    k_w_mk: float
    is_fan: bool = False
    # Whether this component blocks 2D top-down momentum. Components
    # physically below the air plane (logic board, batteries, speakers,
    # ports) emit heat but should not obstruct flow in the 2D model.
    blocks_flow: bool = True
    # Optional per-component "effective slab thickness" in meters used in
    # the 1D out-of-plane closure: q_volumetric = q_w / (area_2d * depth_m).
    # If None, falls back to domain.depth.{active,battery}_zone_m by zone.
    depth_m: float | None = None

    @property
    def area_m2(self) -> float:
        x0, x1 = self.x_range_m
        y0, y1 = self.y_range_m
        return (x1 - x0) * (y1 - y0)

    def overlaps(self, other: "Component") -> bool:
        ax0, ax1 = self.x_range_m
        ay0, ay1 = self.y_range_m
        bx0, bx1 = other.x_range_m
        by0, by1 = other.y_range_m
        return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


@dataclass(frozen=True)
class DepthClosure:
    active_zone_m: float
    battery_zone_m: float
    zone_split_y_m: float


@dataclass(frozen=True)
class Domain:
    length_m: float
    height_m: float
    nx: int
    ny: int
    depth: DepthClosure


@dataclass(frozen=True)
class Fluid:
    rho_kg_m3: float
    mu_pa_s: float
    k_air_w_mk: float
    cp_j_kgk: float
    k_chassis_battery_w_mk: float
    # Effective bulk k in the active zone (metal frame, brackets, board
    # foil, etc. that surround the chips). Higher than k_air to model the
    # actual heat-spreading structure that 2D-top-down would otherwise miss.
    k_chassis_active_w_mk: float = 0.028

    @property
    def nu_m2_s(self) -> float:
        return self.mu_pa_s / self.rho_kg_m3


@dataclass(frozen=True)
class BoundaryConditions:
    wall_temperature_c: float
    inlet_temperature_c: float
    inlet_velocity_m_s: float
    inlet_y_m: float
    inlet_x_range_m: Tuple[float, float]
    outlet_bands_x_m: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class Turbulence:
    model: str
    mixing_length_m: float
    nu_t_max_ratio: float


@dataclass(frozen=True)
class SolverParams:
    max_iterations: int
    tolerance: float
    cfl: float
    pressure_relax: float
    momentum_relax: float
    brinkman_penalty_per_s: float
    log_every: int


@dataclass(frozen=True)
class ValidationRanges:
    soc_mean_c: Tuple[float, float]
    battery_max_c: Tuple[float, float]
    palm_mean_c: Tuple[float, float]
    exhaust_mean_c: Tuple[float, float]
    palm_x_m: Tuple[float, float]
    palm_y_m: Tuple[float, float]


@dataclass(frozen=True)
class Config:
    domain: Domain
    fluid: Fluid
    bc: BoundaryConditions
    fan_velocity_m_s: float
    turbulence: Turbulence
    solver: SolverParams
    components: Tuple[Component, ...]
    validation: ValidationRanges
    source_path: Path | None = field(default=None, compare=False)


def _to_pair(seq) -> Tuple[float, float]:
    a, b = seq
    return float(a), float(b)


def _component_from_dict(raw: dict) -> Component:
    return Component(
        name=str(raw["name"]),
        x_range_m=tuple(v * 1e-3 for v in raw["x_mm"]),  # type: ignore[arg-type]
        y_range_m=tuple(v * 1e-3 for v in raw["y_mm"]),  # type: ignore[arg-type]
        q_watts=float(raw["q_w"]),
        k_w_mk=float(raw["k_w_mk"]),
        is_fan=bool(raw.get("is_fan", False)),
        blocks_flow=bool(raw.get("blocks_flow", True)),
        depth_m=float(raw["depth_m"]) if "depth_m" in raw else None,
    )


def _validate(config: Config) -> None:
    L, H = config.domain.length_m, config.domain.height_m

    if config.domain.nx < 16 or config.domain.ny < 16:
        raise ValueError("Grid too coarse: nx, ny must be >= 16")
    if not (0 < config.bc.inlet_y_m < H):
        raise ValueError(f"inlet_y_m={config.bc.inlet_y_m} outside (0, {H})")

    ix0, ix1 = config.bc.inlet_x_range_m
    if not (0 <= ix0 < ix1 <= L):
        raise ValueError(f"inlet_x_range_m {config.bc.inlet_x_range_m} outside [0, {L}]")

    for ox0, ox1 in config.bc.outlet_bands_x_m:
        if not (0 <= ox0 < ox1 <= L):
            raise ValueError(f"outlet band [{ox0}, {ox1}] outside [0, {L}]")

    for c in config.components:
        x0, x1 = c.x_range_m
        y0, y1 = c.y_range_m
        if not (0 <= x0 < x1 <= L and 0 <= y0 < y1 <= H):
            raise ValueError(f"Component {c.name} bbox out of domain")

    overlaps = []
    comps = config.components
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if comps[i].overlaps(comps[j]):
                overlaps.append((comps[i].name, comps[j].name))
    if overlaps:
        raise ValueError(f"Component overlaps: {overlaps}")


def load(path: str | Path) -> Config:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    d = raw["domain"]
    depth = DepthClosure(
        active_zone_m=float(d["depth_closure"]["active_zone_m"]),
        battery_zone_m=float(d["depth_closure"]["battery_zone_m"]),
        zone_split_y_m=float(d["depth_closure"]["zone_split_y_m"]),
    )
    domain = Domain(
        length_m=float(d["length_m"]),
        height_m=float(d["height_m"]),
        nx=int(d["nx"]),
        ny=int(d["ny"]),
        depth=depth,
    )

    f = raw["fluid"]
    fluid = Fluid(
        rho_kg_m3=float(f["rho_kg_m3"]),
        mu_pa_s=float(f["mu_pa_s"]),
        k_air_w_mk=float(f["k_air_w_mk"]),
        cp_j_kgk=float(f["cp_j_kgk"]),
        k_chassis_battery_w_mk=float(f["k_chassis_battery_w_mk"]),
        k_chassis_active_w_mk=float(f.get("k_chassis_active_w_mk", f["k_air_w_mk"])),
    )

    b = raw["boundary_conditions"]
    bc = BoundaryConditions(
        wall_temperature_c=float(b["wall_temperature_c"]),
        inlet_temperature_c=float(b["inlet_temperature_c"]),
        inlet_velocity_m_s=float(b["inlet_velocity_m_s"]),
        inlet_y_m=float(b["inlet_y_m"]),
        inlet_x_range_m=_to_pair(b["inlet_x_range_m"]),
        outlet_bands_x_m=tuple(_to_pair(p) for p in b["outlet_bands_x_m"]),
    )

    t = raw["turbulence"]
    turbulence = Turbulence(
        model=str(t["model"]),
        mixing_length_m=float(t["mixing_length_m"]),
        nu_t_max_ratio=float(t["nu_t_max_ratio"]),
    )

    s = raw["solver"]
    solver = SolverParams(
        max_iterations=int(s["max_iterations"]),
        tolerance=float(s["tolerance"]),
        cfl=float(s["cfl"]),
        pressure_relax=float(s["pressure_relax"]),
        momentum_relax=float(s["momentum_relax"]),
        brinkman_penalty_per_s=float(s["brinkman_penalty_per_s"]),
        log_every=int(s["log_every"]),
    )

    components = tuple(_component_from_dict(c) for c in raw["components"])

    v = raw["validation"]
    validation = ValidationRanges(
        soc_mean_c=_to_pair(v["soc_mean_c"]),
        battery_max_c=_to_pair(v["battery_max_c"]),
        palm_mean_c=_to_pair(v["palm_mean_c"]),
        exhaust_mean_c=_to_pair(v["exhaust_mean_c"]),
        palm_x_m=_to_pair(v["palm_rest_region"]["x_m"]),
        palm_y_m=_to_pair(v["palm_rest_region"]["y_m"]),
    )

    cfg = Config(
        domain=domain,
        fluid=fluid,
        bc=bc,
        fan_velocity_m_s=float(raw["fans"]["velocity_m_s"]),
        turbulence=turbulence,
        solver=solver,
        components=components,
        validation=validation,
        source_path=path.resolve(),
    )
    _validate(cfg)
    return cfg
