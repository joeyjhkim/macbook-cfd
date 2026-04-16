"""Algebraic Prandtl mixing-length turbulence closure.

For a 2D incompressible flow on a staggered MAC grid, returns the eddy
viscosity field on cell centers given face velocities. The model is
intentionally simple — it is not a substitute for k–omega/k–epsilon, but
captures first-order mixing in the fan jet region without iterations.
"""

from __future__ import annotations

import numpy as np


def eddy_viscosity_cell(
    u: np.ndarray,        # (ny, nx+1)
    v: np.ndarray,        # (ny+1, nx)
    dx: float,
    dy: float,
    mixing_length_m: float,
) -> np.ndarray:
    """Return ν_t on cell centers, shape (ny, nx). Zero if mixing_length=0."""
    if mixing_length_m <= 0:
        return np.zeros((u.shape[0], v.shape[1]), dtype=float)

    # Strain rate components on cell centers.
    dudx = (u[:, 1:] - u[:, :-1]) / dx          # (ny, nx)
    dvdy = (v[1:, :] - v[:-1, :]) / dy          # (ny, nx)

    # ∂u/∂y on cell centers: average ∂u/∂y from two adjacent u-faces.
    # u shape (ny, nx+1); ∂u/∂y at (j+0.5, i) face ≈ (u[j+1,i] - u[j,i])/dy.
    dudy_uface = np.zeros_like(u)               # (ny, nx+1) padded
    dudy_uface[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
    dudy_uface[0, :]  = (u[1, :]  - u[0, :])  / dy
    dudy_uface[-1, :] = (u[-1, :] - u[-2, :]) / dy
    dudy = 0.5 * (dudy_uface[:, :-1] + dudy_uface[:, 1:])  # (ny, nx)

    dvdx_vface = np.zeros_like(v)               # (ny+1, nx)
    dvdx_vface[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
    dvdx_vface[:, 0]  = (v[:, 1]  - v[:, 0])  / dx
    dvdx_vface[:, -1] = (v[:, -1] - v[:, -2]) / dx
    dvdx = 0.5 * (dvdx_vface[:-1, :] + dvdx_vface[1:, :])  # (ny, nx)

    s_mag = np.sqrt(2 * dudx**2 + 2 * dvdy**2 + (dudy + dvdx) ** 2)
    return (mixing_length_m ** 2) * s_mag
