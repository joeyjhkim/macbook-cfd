"""3D steady convection–diffusion energy equation.

Cell-centered finite volume; harmonic-mean conductivity on each face
(handles high-k metal abutting low-k air), first-order upwind for
advection (Peclet-robust).

Boundary conditions:
  - bottom wall (z = 0): Dirichlet T = bottom_wall_temperature_c
  - top (z = Lz), side walls (x = 0, x = Lx), front (y = 0): adiabatic
  - rear outlet patches on y = Ly: zero-gradient (Neumann)
  - rear non-outlet regions on y = Ly: adiabatic
  - inlet bottom face cells: Dirichlet T = inlet.temperature_c
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .config3d import Config3D
from .flow3d import FlowField3D
from .geometry3d import Geometry3D


@dataclass
class TemperatureField3D:
    T: np.ndarray   # (nz, ny, nx), Celsius


def _harmonic_face(k: np.ndarray, axis: int) -> np.ndarray:
    if axis == 2:
        a, b = k[:, :, :-1], k[:, :, 1:]
    elif axis == 1:
        a, b = k[:, :-1, :], k[:, 1:, :]
    else:
        a, b = k[:-1, :, :], k[1:, :, :]
    return 2.0 * a * b / (a + b)


def solve(config: Config3D, geom: Geometry3D, flow: FlowField3D) -> TemperatureField3D:
    nx, ny, nz = geom.nx, geom.ny, geom.nz
    dx, dy, dz = geom.dx, geom.dy, geom.dz
    rho = config.fluid.rho_kg_m3
    cp = config.fluid.cp_j_kgk
    T_in = config.bc.inlet.temperature_c
    T_wall = config.bc.bottom_wall_temperature_c

    kx = _harmonic_face(geom.k_field, axis=2)   # (nz, ny, nx-1)
    ky = _harmonic_face(geom.k_field, axis=1)   # (nz, ny-1, nx)
    kz = _harmonic_face(geom.k_field, axis=0)   # (nz-1, ny, nx)

    De = np.zeros((nz, ny, nx)); De[:, :, :-1] = kx / (dx * dx)
    Dw = np.zeros((nz, ny, nx)); Dw[:, :, 1:]  = kx / (dx * dx)
    Dn = np.zeros((nz, ny, nx)); Dn[:, :-1, :] = ky / (dy * dy)
    Ds = np.zeros((nz, ny, nx)); Ds[:, 1:,  :] = ky / (dy * dy)
    Dt = np.zeros((nz, ny, nx)); Dt[:-1, :, :] = kz / (dz * dz)
    Db = np.zeros((nz, ny, nx)); Db[1:,  :, :] = kz / (dz * dz)

    # Bottom wall Dirichlet: ghost half-cell at distance dz/2.
    Db[0, :, :] = 2.0 * geom.k_field[0, :, :] / (dz * dz)

    u_c, v_c, w_c = flow.u_cell, flow.v_cell, flow.w_cell
    Fw = rho * cp * np.maximum( u_c, 0.0) / dx
    Fe = rho * cp * np.maximum(-u_c, 0.0) / dx
    Fs = rho * cp * np.maximum( v_c, 0.0) / dy
    Fn = rho * cp * np.maximum(-v_c, 0.0) / dy
    Fb = rho * cp * np.maximum( w_c, 0.0) / dz
    Ft = rho * cp * np.maximum(-w_c, 0.0) / dz
    Fw[:, :, 0]  = 0.0
    Fe[:, :, -1] = 0.0
    Fs[:, 0,  :] = 0.0
    Fn[:, -1, :] = 0.0
    Fb[0,  :, :] = 0.0
    Ft[-1, :, :] = 0.0

    n = nx * ny * nz
    idx = np.arange(n).reshape(nz, ny, nx)

    rows: list = []
    cols: list = []
    data: list = []
    rhs = np.zeros(n)

    diag = -(De + Dw + Dn + Ds + Dt + Db + Fe + Fw + Fn + Fs + Fb + Ft)

    def push(mask, off, coeff):
        m = mask.ravel()
        rows.append(idx.ravel()[m])
        cols.append((idx + off).ravel()[m])
        data.append(coeff.ravel()[m])

    east  = np.zeros((nz, ny, nx), dtype=bool); east[:,  :, :-1] = True
    west  = np.zeros((nz, ny, nx), dtype=bool); west[:,  :, 1:]  = True
    north = np.zeros((nz, ny, nx), dtype=bool); north[:, :-1, :] = True
    south = np.zeros((nz, ny, nx), dtype=bool); south[:, 1:,  :] = True
    top   = np.zeros((nz, ny, nx), dtype=bool); top[:-1,   :, :] = True
    bot   = np.zeros((nz, ny, nx), dtype=bool); bot[1:,    :, :] = True

    push(east,  +1,      De)
    push(west,  -1,      Dw)
    push(north, +nx,     Dn)
    push(south, -nx,     Ds)
    push(top,   +nx * ny, Dt)
    push(bot,   -nx * ny, Db)

    push(east,  +1,      Fe)
    push(west,  -1,      Fw)
    push(north, +nx,     Fn)
    push(south, -nx,     Fs)
    push(top,   +nx * ny, Ft)
    push(bot,   -nx * ny, Fb)

    rows.append(idx.ravel()); cols.append(idx.ravel()); data.append(diag.ravel())

    rhs -= geom.q_volumetric.ravel()
    # Bottom wall contribution: T_wall on the ghost side.
    rhs[idx[0, :, :].ravel()] -= (Db[0, :, :] * T_wall).ravel()

    A = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tolil()

    # Inlet Dirichlet pin on the bottom slab (z=0) cells inside inlet mask.
    inlet_cells = idx[0, geom.inlet_bottom_mask]
    for k in inlet_cells.ravel():
        A.rows[int(k)] = [int(k)]
        A.data[int(k)] = [1.0]
        rhs[int(k)] = T_in

    A = A.tocsc()
    lu = splu(A)
    T_flat = lu.solve(rhs)
    T = T_flat.reshape(nz, ny, nx)
    T = np.clip(T, T_in - 1.0, None)
    return TemperatureField3D(T=T)
