"""3D configuration loading for the MacBook case.

Unlike the 2D loader, each component carries an explicit z-range so the
out-of-plane dimension is resolved rather than lumped into a depth
closure. The inlet and outlet boundary conditions are genuinely 3D: the
inlet is a patch on the bottom face (z=0), outlets are patches on the
rear face (y=H_y).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import yaml


@dataclass(frozen=True)
class Component3D:
    name: str
    x_range_m: Tuple[float, float]
    y_range_m: Tuple[float, float]
    z_range_m: Tuple[float, float]
    q_watts: float
    k_w_mk: float
    is_fan: bool = False
    fan_direction: str = "+y"     # "+y" = blow toward rear hinge (outlet)
    blocks_flow: bool = True

    @property
    def volume_m3(self) -> float:
        x0, x1 = self.x_range_m
        y0, y1 = self.y_range_m
        z0, z1 = self.z_range_m
        return (x1 - x0) * (y1 - y0) * (z1 - z0)

    def overlaps(self, other: "Component3D") -> bool:
        for ax, bx in (
            (self.x_range_m, other.x_range_m),
            (self.y_range_m, other.y_range_m),
            (self.z_range_m, other.z_range_m),
        ):
            if ax[1] <= bx[0] or bx[1] <= ax[0]:
                return False
        return True


@dataclass(frozen=True)
class Domain3D:
    length_x_m: float
    length_y_m: float
    length_z_m: float
    nx: int
    ny: int
    nz: int


@dataclass(frozen=True)
class Fluid3D:
    rho_kg_m3: float
    mu_pa_s: float
    k_air_w_mk: float
    cp_j_kgk: float
    k_chassis_active_w_mk: float = 0.028
    k_chassis_battery_w_mk: float = 0.028
    chassis_split_z_m: float = 0.010

    @property
    def nu_m2_s(self) -> float:
        return self.mu_pa_s / self.rho_kg_m3


@dataclass(frozen=True)
class InletPatch:
    x_range_m: Tuple[float, float]
    y_range_m: Tuple[float, float]
    velocity_m_s: float
    temperature_c: float


@dataclass(frozen=True)
class OutletPatch:
    x_range_m: Tuple[float, float]
    z_range_m: Tuple[float, float]


@dataclass(frozen=True)
class BoundaryConditions3D:
    bottom_wall_temperature_c: float
    ambient_temperature_c: float
    inlet: InletPatch
    outlets: Tuple[OutletPatch, ...]


@dataclass(frozen=True)
class Turbulence3D:
    model: str
    mixing_length_m: float
    nu_t_max_ratio: float


@dataclass(frozen=True)
class SolverParams3D:
    max_iterations: int
    tolerance: float
    cfl: float
    pressure_relax: float
    momentum_relax: float
    brinkman_penalty_per_s: float
    log_every: int


@dataclass(frozen=True)
class ValidationRanges3D:
    soc_mean_c: Tuple[float, float]
    battery_max_c: Tuple[float, float]
    palm_mean_c: Tuple[float, float]
    exhaust_mean_c: Tuple[float, float]
    palm_region_x_m: Tuple[float, float]
    palm_region_y_m: Tuple[float, float]


@dataclass(frozen=True)
class Config3D:
    domain: Domain3D
    fluid: Fluid3D
    bc: BoundaryConditions3D
    fan_velocity_m_s: float
    turbulence: Turbulence3D
    solver: SolverParams3D
    components: Tuple[Component3D, ...]
    validation: ValidationRanges3D
    source_path: Path | None = field(default=None, compare=False)


def _triple(seq) -> Tuple[float, float]:
    a, b = seq
    return float(a), float(b)


def _component_from_dict(raw: dict) -> Component3D:
    return Component3D(
        name=str(raw["name"]),
        x_range_m=tuple(v * 1e-3 for v in raw["x_mm"]),  # type: ignore[arg-type]
        y_range_m=tuple(v * 1e-3 for v in raw["y_mm"]),  # type: ignore[arg-type]
        z_range_m=tuple(v * 1e-3 for v in raw["z_mm"]),  # type: ignore[arg-type]
        q_watts=float(raw["q_w"]),
        k_w_mk=float(raw["k_w_mk"]),
        is_fan=bool(raw.get("is_fan", False)),
        fan_direction=str(raw.get("fan_direction", "+y")),
        blocks_flow=bool(raw.get("blocks_flow", True)),
    )


def _validate(config: Config3D) -> None:
    Lx, Ly, Lz = (
        config.domain.length_x_m, config.domain.length_y_m, config.domain.length_z_m
    )
    d = config.domain
    if d.nx < 16 or d.ny < 16 or d.nz < 8:
        raise ValueError("Grid too coarse: nx, ny >= 16 and nz >= 8")
    for c in config.components:
        x0, x1 = c.x_range_m
        y0, y1 = c.y_range_m
        z0, z1 = c.z_range_m
        if not (0 <= x0 < x1 <= Lx and 0 <= y0 < y1 <= Ly and 0 <= z0 < z1 <= Lz):
            raise ValueError(f"Component {c.name} bbox out of domain")
        if c.is_fan and c.fan_direction not in ("+y", "-y", "+x", "-x"):
            raise ValueError(f"Fan {c.name} direction must be ±x or ±y")
    overlaps = []
    comps = config.components
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if comps[i].overlaps(comps[j]):
                overlaps.append((comps[i].name, comps[j].name))
    if overlaps:
        raise ValueError(f"Component overlaps: {overlaps}")


def load(path: str | Path) -> Config3D:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    d = raw["domain"]
    domain = Domain3D(
        length_x_m=float(d["length_x_m"]),
        length_y_m=float(d["length_y_m"]),
        length_z_m=float(d["length_z_m"]),
        nx=int(d["nx"]), ny=int(d["ny"]), nz=int(d["nz"]),
    )

    fl = raw["fluid"]
    fluid = Fluid3D(
        rho_kg_m3=float(fl["rho_kg_m3"]),
        mu_pa_s=float(fl["mu_pa_s"]),
        k_air_w_mk=float(fl["k_air_w_mk"]),
        cp_j_kgk=float(fl["cp_j_kgk"]),
        k_chassis_active_w_mk=float(fl.get("k_chassis_active_w_mk", fl["k_air_w_mk"])),
        k_chassis_battery_w_mk=float(fl.get("k_chassis_battery_w_mk", fl["k_air_w_mk"])),
        chassis_split_z_m=float(fl.get("chassis_split_z_m", 0.010)),
    )

    b = raw["boundary_conditions"]
    inlet_raw = b["inlet"]
    inlet = InletPatch(
        x_range_m=_triple(inlet_raw["x_m"]),
        y_range_m=_triple(inlet_raw["y_m"]),
        velocity_m_s=float(inlet_raw["velocity_m_s"]),
        temperature_c=float(inlet_raw["temperature_c"]),
    )
    outlets = tuple(
        OutletPatch(
            x_range_m=_triple(o["x_m"]),
            z_range_m=_triple(o["z_m"]),
        )
        for o in b["outlets"]
    )
    bc = BoundaryConditions3D(
        bottom_wall_temperature_c=float(b["bottom_wall_temperature_c"]),
        ambient_temperature_c=float(b["ambient_temperature_c"]),
        inlet=inlet,
        outlets=outlets,
    )

    t = raw["turbulence"]
    turbulence = Turbulence3D(
        model=str(t["model"]),
        mixing_length_m=float(t["mixing_length_m"]),
        nu_t_max_ratio=float(t["nu_t_max_ratio"]),
    )

    s = raw["solver"]
    solver = SolverParams3D(
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
    validation = ValidationRanges3D(
        soc_mean_c=_triple(v["soc_mean_c"]),
        battery_max_c=_triple(v["battery_max_c"]),
        palm_mean_c=_triple(v["palm_mean_c"]),
        exhaust_mean_c=_triple(v["exhaust_mean_c"]),
        palm_region_x_m=_triple(v["palm_rest_region"]["x_m"]),
        palm_region_y_m=_triple(v["palm_rest_region"]["y_m"]),
    )

    cfg = Config3D(
        domain=domain, fluid=fluid, bc=bc,
        fan_velocity_m_s=float(raw["fans"]["velocity_m_s"]),
        turbulence=turbulence, solver=solver,
        components=components, validation=validation,
        source_path=path.resolve(),
    )
    _validate(cfg)
    return cfg
