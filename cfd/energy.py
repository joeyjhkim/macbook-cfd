"""Steady-state convection–diffusion energy equation.

Cell-centered grid (the same one used for pressure/temperature in
`Geometry`). Convective term uses first-order upwind for diagonal
dominance; diffusive flux uses harmonic-mean conductivity on faces so
that high-k metal abuts low-k air without Gibbs-style artifacts.

Boundary conditions:
  - bottom wall: Dirichlet T = wall_temperature_c
  - top, sides : adiabatic (Neumann)
  - inlet row  : Dirichlet T = inlet_temperature_c
  - outlets    : zero-gradient (handled via Neumann implicitly)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .config import Config
from .geometry import Geometry
from .flow import FlowField


@dataclass
class TemperatureField:
    T: np.ndarray  # (ny, nx), Celsius


def _harmonic_face(k: np.ndarray, axis: int) -> np.ndarray:
    """Harmonic-mean conductivity on faces between adjacent cells along axis."""
    if axis == 1:  # x-faces between (j, i) and (j, i+1) — shape (ny, nx-1)
        a, b = k[:, :-1], k[:, 1:]
    else:           # y-faces between (j, i) and (j+1, i) — shape (ny-1, nx)
        a, b = k[:-1, :], k[1:, :]
    return 2.0 * a * b / (a + b)


def solve(config: Config, geom: Geometry, flow: FlowField) -> TemperatureField:
    nx, ny, dx, dy = geom.nx, geom.ny, geom.dx, geom.dy
    rho = config.fluid.rho_kg_m3
    cp = config.fluid.cp_j_kgk
    T_in = config.bc.inlet_temperature_c
    T_wall = config.bc.wall_temperature_c

    # Face conductivities (harmonic mean)
    kx_face = _harmonic_face(geom.k_field, axis=1)   # (ny, nx-1)
    ky_face = _harmonic_face(geom.k_field, axis=0)   # (ny-1, nx)

    # Diffusion coefficients per cell from each neighbor
    De = np.zeros((ny, nx)); De[:, :-1] = kx_face / (dx * dx)   # east face
    Dw = np.zeros((ny, nx)); Dw[:, 1:]  = kx_face / (dx * dx)   # west face
    Dn = np.zeros((ny, nx)); Dn[:-1, :] = ky_face / (dy * dy)   # north face
    Ds = np.zeros((ny, nx)); Ds[1:, :]  = ky_face / (dy * dy)   # south face

    # Bottom wall Dirichlet: ghost cell at distance dy/2; effective coefficient 2k/dy²
    Ds[0, :] = 2.0 * geom.k_field[0, :] / (dy * dy)

    # Convective coefficients (upwind), volumetric: ρ c_p u / dx with u at cell center
    u_c, v_c = flow.u_cell, flow.v_cell
    Fw = rho * cp * np.maximum( u_c, 0.0) / dx     # west face: positive u brings T from west
    Fe = rho * cp * np.maximum(-u_c, 0.0) / dx
    Fs = rho * cp * np.maximum( v_c, 0.0) / dy
    Fn = rho * cp * np.maximum(-v_c, 0.0) / dy
    # Disable convection at boundaries (no upstream cell)
    Fw[:, 0] = 0.0
    Fe[:, -1] = 0.0
    Fs[0, :] = 0.0
    Fn[-1, :] = 0.0

    # Cell-index utilities
    n = nx * ny
    idx = np.arange(n).reshape(ny, nx)

    rows, cols, data = [], [], []
    rhs = np.zeros(n)

    diag = -(De + Dw + Dn + Ds + Fe + Fw + Fn + Fs)

    # Off-diagonals
    def push(mask, off, coeff):
        m = mask.ravel()
        rows.append(idx.ravel()[m])
        cols.append((idx + off).ravel()[m])
        data.append(coeff.ravel()[m])

    east = np.zeros((ny, nx), dtype=bool); east[:, :-1] = True
    west = np.zeros((ny, nx), dtype=bool); west[:, 1:]  = True
    north = np.zeros((ny, nx), dtype=bool); north[:-1, :] = True
    south = np.zeros((ny, nx), dtype=bool); south[1:, :]  = True

    # Diffusion off-diagonals
    push(east,  +1,    De)
    push(west,  -1,    Dw)
    push(north, +nx,   Dn)
    push(south, -nx,   Ds)
    # Convective off-diagonals (advection brings T from upstream cell)
    push(east,  +1,    Fe)
    push(west,  -1,    Fw)
    push(north, +nx,   Fn)
    push(south, -nx,   Fs)

    # Diagonal
    rows.append(idx.ravel()); cols.append(idx.ravel()); data.append(diag.ravel())

    # RHS: heat sources + Dirichlet contributions
    rhs -= geom.q_volumetric.ravel()             # source on RHS with sign of conservation
    rhs[idx[0, :]] -= Ds[0, :] * T_wall          # bottom wall Dirichlet contribution

    A = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tolil()

    # Inlet row Dirichlet pin: T = T_in for cells in inlet_face_mask at j=j_inlet
    inlet_cells = idx[geom.j_inlet, geom.inlet_face_mask]
    for k in inlet_cells:
        A.rows[k] = [k]
        A.data[k] = [1.0]
        rhs[k] = T_in

    A = A.tocsc()
    T_flat = spsolve(A, rhs)
    T = T_flat.reshape(ny, nx)
    T = np.clip(T, T_in - 1.0, None)

    return TemperatureField(T=T)
