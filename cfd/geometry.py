"""Discretization geometry: staggered MAC grid, masks, source fields.

Grid layout (cell-centered MAC):

  - p, T   on cell centers,        shape (ny, nx),    at ((i+0.5)dx, (j+0.5)dy)
  - u      on vertical (x-) faces, shape (ny, nx+1),  at (i*dx, (j+0.5)dy)
  - v      on horizontal y-faces,  shape (ny+1, nx),  at ((i+0.5)dx, j*dy)

The flow domain is restricted to rows above the inlet line (the inlet
represents the bottom-case vents, so the battery half is treated as
conduction-only — heat sources still apply, but no air flow there).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass(frozen=True)
class Geometry:
    nx: int
    ny: int
    dx: float
    dy: float

    x_c: np.ndarray  # cell-center x, shape (nx,)
    y_c: np.ndarray  # cell-center y, shape (ny,)
    x_u: np.ndarray  # u-face x,      shape (nx+1,)
    y_v: np.ndarray  # v-face y,      shape (ny+1,)
    X_c: np.ndarray  # meshgrid (ny, nx)
    Y_c: np.ndarray  # meshgrid (ny, nx)

    is_solid: np.ndarray       # (ny, nx) bool — any component (for k/Q purposes)
    is_solid_flow: np.ndarray  # (ny, nx) bool — components that block 2D momentum
    is_fan: np.ndarray         # (ny, nx) bool — fan components
    comp_id: np.ndarray        # (ny, nx) int  — component index, -1 if fluid
    k_field: np.ndarray        # (ny, nx) thermal conductivity W/(m K)
    q_volumetric: np.ndarray   # (ny, nx) volumetric heat source W/m^3

    j_inlet: int           # v-face index of the inlet line
    inlet_face_mask: np.ndarray  # (nx,) bool — which i-cells in inlet x-range
    outlet_top_mask: np.ndarray  # (nx,) bool — which i-cells are top exhausts

    # Convenience: cells where momentum is solved (active fluid).
    flow_mask: np.ndarray  # (ny, nx) bool — j >= j_inlet AND not solid


def build(config: Config) -> Geometry:
    nx, ny = config.domain.nx, config.domain.ny
    L, H = config.domain.length_m, config.domain.height_m
    dx, dy = L / nx, H / ny

    x_c = (np.arange(nx) + 0.5) * dx
    y_c = (np.arange(ny) + 0.5) * dy
    x_u = np.arange(nx + 1) * dx
    y_v = np.arange(ny + 1) * dy
    X_c, Y_c = np.meshgrid(x_c, y_c)

    is_solid = np.zeros((ny, nx), dtype=bool)
    is_solid_flow = np.zeros((ny, nx), dtype=bool)
    is_fan = np.zeros((ny, nx), dtype=bool)
    comp_id = -np.ones((ny, nx), dtype=np.int32)

    # Anisotropic effective conductivity in the fluid background:
    #   - battery zone: aluminum chassis dominates (high k)
    #   - active zone : metal frame / mounts spread heat between components
    k_field = np.full((ny, nx), config.fluid.k_chassis_active_w_mk, dtype=float)
    k_field[Y_c < config.domain.depth.zone_split_y_m] = config.fluid.k_chassis_battery_w_mk

    q_volumetric = np.zeros((ny, nx), dtype=float)
    depth = config.domain.depth

    for ci, comp in enumerate(config.components):
        x0, x1 = comp.x_range_m
        y0, y1 = comp.y_range_m
        mask = (X_c >= x0) & (X_c <= x1) & (Y_c >= y0) & (Y_c <= y1)
        is_solid |= mask
        if comp.blocks_flow:
            is_solid_flow |= mask
        comp_id[mask] = ci
        k_field[mask] = comp.k_w_mk
        if comp.is_fan:
            is_fan[mask] = True
        if comp.q_watts > 0:
            # 1D conduction closure for out-of-plane heat removal.
            if comp.depth_m is not None:
                depth_m = comp.depth_m
            elif y1 <= depth.zone_split_y_m:
                depth_m = depth.battery_zone_m
            else:
                depth_m = depth.active_zone_m
            q_volumetric[mask] = comp.q_watts / (comp.area_m2 * depth_m)

    j_inlet = int(round(config.bc.inlet_y_m / dy))
    j_inlet = max(1, min(ny - 1, j_inlet))

    ix0, ix1 = config.bc.inlet_x_range_m
    inlet_face_mask = (x_c >= ix0) & (x_c <= ix1)

    outlet_top_mask = np.zeros(nx, dtype=bool)
    for ox0, ox1 in config.bc.outlet_bands_x_m:
        outlet_top_mask |= (x_c >= ox0) & (x_c <= ox1)

    flow_mask = np.zeros((ny, nx), dtype=bool)
    flow_mask[j_inlet:, :] = True
    flow_mask &= ~is_solid_flow

    return Geometry(
        nx=nx, ny=ny, dx=dx, dy=dy,
        x_c=x_c, y_c=y_c, x_u=x_u, y_v=y_v, X_c=X_c, Y_c=Y_c,
        is_solid=is_solid, is_solid_flow=is_solid_flow, is_fan=is_fan, comp_id=comp_id,
        k_field=k_field, q_volumetric=q_volumetric,
        j_inlet=j_inlet,
        inlet_face_mask=inlet_face_mask,
        outlet_top_mask=outlet_top_mask,
        flow_mask=flow_mask,
    )
