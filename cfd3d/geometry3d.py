"""3D MAC-grid geometry, masks, and source fields.

Layout (cell-centered MAC):
  - p, T  at cell centers, shape (nz, ny, nx)
  - u     on x-faces,       shape (nz, ny,   nx+1)  ← velocity along x
  - v     on y-faces,       shape (nz, ny+1, nx)    ← velocity along y
  - w     on z-faces,       shape (nz+1, ny, nx)    ← velocity along z

Coordinate system:
  - x: left–right across laptop width (side speakers on left/right)
  - y: front–back (front = 0, rear hinge = Ly)
  - z: bottom–top of chassis (bottom cover = 0, top / keyboard = Lz)

Boundary conditions:
  - Inlet: bottom face (z = 0) over a rectangular patch → w = v_in pointing +z
  - Outlets: rear face (y = Ly) over rectangular x,z patches → free outflow
  - All other external faces: no-slip walls
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config3d import Config3D


@dataclass(frozen=True)
class Geometry3D:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float

    x_c: np.ndarray
    y_c: np.ndarray
    z_c: np.ndarray
    X_c: np.ndarray    # (nz, ny, nx)
    Y_c: np.ndarray
    Z_c: np.ndarray

    is_solid: np.ndarray        # (nz, ny, nx) — any component (for k/Q)
    is_solid_flow: np.ndarray   # (nz, ny, nx) — components that block momentum
    is_fan: np.ndarray          # (nz, ny, nx) — fan volumes
    fan_axis: np.ndarray        # (nz, ny, nx) int8: 0=none, 1=+y, 2=-y, 3=+x, 4=-x
    comp_id: np.ndarray         # (nz, ny, nx) int32, -1 = fluid
    k_field: np.ndarray         # (nz, ny, nx)
    q_volumetric: np.ndarray    # (nz, ny, nx) W/m^3

    # Inlet face at z = 0. Mask on (ny, nx).
    inlet_bottom_mask: np.ndarray  # (ny, nx) bool
    # Outlet face at y = Ly. Mask on (nz, nx).
    outlet_rear_mask: np.ndarray   # (nz, nx) bool

    # Flow cells: not solid_flow (air-volume cells).
    flow_mask: np.ndarray       # (nz, ny, nx) bool


_FAN_AXIS_CODE = {"+y": 1, "-y": 2, "+x": 3, "-x": 4}


def build(config: Config3D) -> Geometry3D:
    nx, ny, nz = config.domain.nx, config.domain.ny, config.domain.nz
    Lx, Ly, Lz = (
        config.domain.length_x_m,
        config.domain.length_y_m,
        config.domain.length_z_m,
    )
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    x_c = (np.arange(nx) + 0.5) * dx
    y_c = (np.arange(ny) + 0.5) * dy
    z_c = (np.arange(nz) + 0.5) * dz
    Z_c, Y_c, X_c = np.meshgrid(z_c, y_c, x_c, indexing="ij")

    is_solid = np.zeros((nz, ny, nx), dtype=bool)
    is_solid_flow = np.zeros((nz, ny, nx), dtype=bool)
    is_fan = np.zeros((nz, ny, nx), dtype=bool)
    fan_axis = np.zeros((nz, ny, nx), dtype=np.int8)
    comp_id = -np.ones((nz, ny, nx), dtype=np.int32)

    # Background k: boosted in the active zone (metal frame, PCB, brackets)
    # and battery zone (chassis, heat spreader), same approach as 2D solver.
    k_field = np.full((nz, ny, nx), config.fluid.k_chassis_active_w_mk, dtype=float)
    k_field[Z_c < config.fluid.chassis_split_z_m] = config.fluid.k_chassis_battery_w_mk
    q_volumetric = np.zeros((nz, ny, nx), dtype=float)

    for ci, comp in enumerate(config.components):
        x0, x1 = comp.x_range_m
        y0, y1 = comp.y_range_m
        z0, z1 = comp.z_range_m
        mask = (
            (X_c >= x0) & (X_c <= x1)
            & (Y_c >= y0) & (Y_c <= y1)
            & (Z_c >= z0) & (Z_c <= z1)
        )
        is_solid |= mask
        if comp.blocks_flow:
            is_solid_flow |= mask
        comp_id[mask] = ci
        k_field[mask] = comp.k_w_mk
        if comp.is_fan:
            is_fan[mask] = True
            fan_axis[mask] = _FAN_AXIS_CODE[comp.fan_direction]
        if comp.q_watts > 0:
            # Volumetric heat source distributed over the 3D component volume.
            vol = comp.volume_m3
            if vol > 0:
                q_volumetric[mask] = comp.q_watts / vol

    inlet = config.bc.inlet
    ix0, ix1 = inlet.x_range_m
    iy0, iy1 = inlet.y_range_m
    X2, Y2 = np.meshgrid(x_c, y_c)         # (ny, nx)
    inlet_bottom_mask = (
        (X2 >= ix0) & (X2 <= ix1)
        & (Y2 >= iy0) & (Y2 <= iy1)
    )

    outlet_rear_mask = np.zeros((nz, nx), dtype=bool)
    Xz, Zz = np.meshgrid(x_c, z_c)         # (nz, nx)
    for out in config.bc.outlets:
        ox0, ox1 = out.x_range_m
        oz0, oz1 = out.z_range_m
        outlet_rear_mask |= (
            (Xz >= ox0) & (Xz <= ox1)
            & (Zz >= oz0) & (Zz <= oz1)
        )

    # Flow cells: everywhere not blocked by momentum-blocking components.
    flow_mask = ~is_solid_flow

    return Geometry3D(
        nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
        x_c=x_c, y_c=y_c, z_c=z_c, X_c=X_c, Y_c=Y_c, Z_c=Z_c,
        is_solid=is_solid, is_solid_flow=is_solid_flow,
        is_fan=is_fan, fan_axis=fan_axis, comp_id=comp_id,
        k_field=k_field, q_volumetric=q_volumetric,
        inlet_bottom_mask=inlet_bottom_mask,
        outlet_rear_mask=outlet_rear_mask,
        flow_mask=flow_mask,
    )
