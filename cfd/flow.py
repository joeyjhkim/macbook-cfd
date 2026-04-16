"""Incompressible Navier–Stokes on a staggered MAC grid.

Algorithm: pseudo-transient projection (Chorin) with implicit Brinkman
penalization for solid cells, fans, and the inlet face.

Each pseudo-time step:

    1. Momentum predictor with explicit advection + conservative
       variable-viscosity diffusion and implicit Brinkman drag (so any
       cell can be pinned to a target velocity without shrinking dt).
    2. Pressure Poisson solved on flow_mask cells; Dirichlet dp=0 at
       outlet bands, Neumann at every other boundary face.
    3. Velocity correction from grad(dp) — this step is mass-conserving
       and is NOT under-relaxed (would destroy divergence-freeness).
    4. Pressure field is under-relaxed (alpha_p) — separate from the
       velocity, because only p (not u) accumulates.

Convergence metric: L∞ of divergence on flow cells. Target ~1e-6 * u_ref/h.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from .config import Config
from .geometry import Geometry
from .turbulence import eddy_viscosity_cell


@dataclass
class FlowField:
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    u_cell: np.ndarray
    v_cell: np.ndarray
    speed_cell: np.ndarray
    vorticity_cell: np.ndarray
    iterations: int
    final_residual: float
    final_momentum_res: float
    converged: bool
    div_history: list
    mom_history: list


def _build_pressure_solver(geom: Geometry):
    """Assemble and factorize the pressure-correction Poisson matrix."""
    nx, ny, dx, dy = geom.nx, geom.ny, geom.dx, geom.dy
    inv_dx2, inv_dy2 = 1.0 / (dx * dx), 1.0 / (dy * dy)

    flow = geom.flow_mask
    cell_idx = -np.ones((ny, nx), dtype=np.int64)
    flat = np.flatnonzero(flow.ravel())
    cell_idx.ravel()[flat] = np.arange(flat.size)
    n_unknowns = flat.size

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    diag_arr = np.zeros(n_unknowns, dtype=float)
    has_dirichlet = np.zeros(n_unknowns, dtype=bool)

    js, is_ = np.where(flow)
    for k, (j, i) in enumerate(zip(js.tolist(), is_.tolist())):
        if i - 1 >= 0 and flow[j, i - 1]:
            rows.append(k); cols.append(int(cell_idx[j, i - 1])); data.append(inv_dx2)
            diag_arr[k] -= inv_dx2
        if i + 1 < nx and flow[j, i + 1]:
            rows.append(k); cols.append(int(cell_idx[j, i + 1])); data.append(inv_dx2)
            diag_arr[k] -= inv_dx2
        if j - 1 >= 0 and flow[j - 1, i]:
            rows.append(k); cols.append(int(cell_idx[j - 1, i])); data.append(inv_dy2)
            diag_arr[k] -= inv_dy2
        if j + 1 < ny and flow[j + 1, i]:
            rows.append(k); cols.append(int(cell_idx[j + 1, i])); data.append(inv_dy2)
            diag_arr[k] -= inv_dy2
        elif j == ny - 1 and geom.outlet_top_mask[i]:
            diag_arr[k] -= inv_dy2
            has_dirichlet[k] = True

    if n_unknowns:
        adj = csr_matrix(
            (np.ones(len(rows), dtype=float), (rows, cols)),
            shape=(n_unknowns, n_unknowns),
        )
        n_comp, comp_labels = connected_components(adj, directed=False)
        for c in range(n_comp):
            members = np.where(comp_labels == c)[0]
            if not has_dirichlet[members].any():
                pin = int(members[0])
                keep = [r != pin for r in rows]
                rows[:] = [r for r, k_ in zip(rows, keep) if k_]
                cols[:] = [c_ for c_, k_ in zip(cols, keep) if k_]
                data[:] = [d for d, k_ in zip(data, keep) if k_]
                diag_arr[pin] = 1.0

    isolated = (diag_arr == 0.0)
    if isolated.any():
        diag_arr[isolated] = 1.0

    rows.extend(range(n_unknowns))
    cols.extend(range(n_unknowns))
    data.extend(diag_arr.tolist())

    A = csc_matrix((data, (rows, cols)), shape=(n_unknowns, n_unknowns))
    lu = splu(A)

    pinned = isolated.copy()
    return lu, cell_idx, n_unknowns, pinned


def _convect_u(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """First-order upwind convective term for u-momentum, shape (ny, nx+1)."""
    ny, nx_p1 = u.shape
    conv = np.zeros_like(u)

    u_c = u[:, 1:-1]                                   # (ny, nx-1)
    du_dx_back = (u[:, 1:-1] - u[:, :-2]) / dx
    du_dx_fwd  = (u[:, 2:]  - u[:, 1:-1]) / dx
    pos = u_c > 0
    du_dx = np.where(pos, du_dx_back, du_dx_fwd)

    # v interpolated to interior u-face columns i=1..nx-1:
    #   corner average of v[j, i-1], v[j, i], v[j+1, i-1], v[j+1, i]
    v_left  = v[:, :-1]
    v_right = v[:, 1:]
    v_at_uface_int = 0.25 * (v_left[:-1] + v_left[1:] + v_right[:-1] + v_right[1:])

    u_pad = np.zeros((ny + 2, nx_p1), dtype=u.dtype)
    u_pad[1:-1, :] = u
    u_pad[0, :]  = -u[0, :]
    u_pad[-1, :] = -u[-1, :]
    du_dy_back = (u_pad[1:-1, 1:-1] - u_pad[:-2, 1:-1]) / dy
    du_dy_fwd  = (u_pad[2:, 1:-1]  - u_pad[1:-1, 1:-1]) / dy
    pos_v = v_at_uface_int > 0
    du_dy = np.where(pos_v, du_dy_back, du_dy_fwd)

    conv[:, 1:-1] = u_c * du_dx + v_at_uface_int * du_dy
    return conv


def _convect_v(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    ny_p1, nx = v.shape
    conv = np.zeros_like(v)

    v_c = v[1:-1, :]
    dv_dy_back = (v[1:-1, :] - v[:-2, :]) / dy
    dv_dy_fwd  = (v[2:, :]  - v[1:-1, :]) / dy
    pos = v_c > 0
    dv_dy = np.where(pos, dv_dy_back, dv_dy_fwd)

    u_top    = u[:-1, :]
    u_bottom = u[1:,  :]
    u_at_vface_int = 0.25 * (u_top[:, :-1] + u_top[:, 1:] + u_bottom[:, :-1] + u_bottom[:, 1:])

    v_pad = np.zeros((ny_p1, nx + 2), dtype=v.dtype)
    v_pad[:, 1:-1] = v
    v_pad[:, 0]  = -v[:, 0]
    v_pad[:, -1] = -v[:, -1]
    dv_dx_back = (v_pad[1:-1, 1:-1] - v_pad[1:-1, :-2]) / dx
    dv_dx_fwd  = (v_pad[1:-1, 2:]  - v_pad[1:-1, 1:-1]) / dx
    pos_u = u_at_vface_int > 0
    dv_dx = np.where(pos_u, dv_dx_back, dv_dx_fwd)

    conv[1:-1, :] = u_at_vface_int * dv_dx + v_c * dv_dy
    return conv


def _diff_u(u: np.ndarray, nu_cell: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Conservative variable-viscosity diffusion on u-momentum.

    Computes   D_u = (1/dx) d/dx(nu du/dx) + (1/dy) d/dy(nu du/dy)
    with nu evaluated at the faces of the u-control volume:
      - east/west fluxes: nu at cell centers (nu_cell[j, i])
      - north/south fluxes: nu at cell corners (average of 4 neighbors)
    Returns array shape matching u (ny, nx+1).
    """
    ny, nx_p1 = u.shape
    nx = nx_p1 - 1
    out = np.zeros_like(u)

    # Pad u with no-slip ghosts in y for the south/north terms.
    u_pad = np.zeros((ny + 2, nx_p1), dtype=u.dtype)
    u_pad[1:-1, :] = u
    u_pad[0,  :] = -u[0,  :]
    u_pad[-1, :] = -u[-1, :]

    # ---- x-direction: d/dx (nu du/dx) ----
    # For u-face (j, i), i in 1..nx-1, the control volume spans cells i-1..i.
    # Flux across east face of CV is at cell center i:  nu_cell[j, i] * (u[j, i+1] - u[j, i])/dx
    # Flux across west face of CV is at cell center i-1: nu_cell[j, i-1] * (u[j, i] - u[j, i-1])/dx
    flux_e = nu_cell[:, :] * (u[:, 1:] - u[:, :-1]) / dx      # (ny, nx)
    # flux_e at column i corresponds to CV east face when u-face index = i. But our u index runs 0..nx.
    # For interior i=1..nx-1:
    #   east flux at cell column i  (between u[:,i] and u[:,i+1])
    #   west flux at cell column i-1 (between u[:,i-1] and u[:,i])
    out[:, 1:-1] = (flux_e[:, 1:] - flux_e[:, :-1]) / dx      # (ny, nx-1)

    # ---- y-direction: d/dy (nu du/dy) ----
    # CV for u at (j, i) spans cells (i-1, j) and (i, j). Corner conductivity at
    # (j+0.5, i) = avg of nu_cell[j, i-1], nu_cell[j, i], nu_cell[j+1, i-1], nu_cell[j+1, i].
    # Pad nu_cell with Neumann (copy) in x and y for edge columns.
    nu_pad = np.zeros((ny + 2, nx + 2), dtype=nu_cell.dtype)
    nu_pad[1:-1, 1:-1] = nu_cell
    nu_pad[0,    1:-1] = nu_cell[0, :]
    nu_pad[-1,   1:-1] = nu_cell[-1, :]
    nu_pad[1:-1, 0]    = nu_cell[:, 0]
    nu_pad[1:-1, -1]   = nu_cell[:, -1]
    nu_pad[0,    0]    = nu_cell[0, 0]
    nu_pad[0,   -1]    = nu_cell[0, -1]
    nu_pad[-1,   0]    = nu_cell[-1, 0]
    nu_pad[-1,  -1]    = nu_cell[-1, -1]

    # Corner nu at top-right corner of cell (j, i) — index [j+1, i+1] in nu_pad.
    # For u-face (j, i) with i in [1, nx-1]:
    #   north corner between rows j and j+1: avg nu_pad[j, i-1..i] and nu_pad[j+1, i-1..i] ... carefully.
    # Simpler: compute corner nu once, shape (ny+1, nx+1), then index.
    # corner[jc, ic] = average of nu_cell of 4 surrounding cells (jc-1..jc, ic-1..ic).
    corner = 0.25 * (nu_pad[:-1, :-1] + nu_pad[1:, :-1] + nu_pad[:-1, 1:] + nu_pad[1:, 1:])  # (ny+1, nx+1)

    # Flux at north face of u-CV at (j, i) — between rows j and j+1 at column i:
    #   nu_corner[j+1, i] * (u[j+1, i] - u[j, i])/dy
    # For j in 0..ny-1, this is:
    flux_n = corner[1:, :] * (u_pad[2:, :] - u_pad[1:-1, :]) / dy       # (ny, nx+1)
    flux_s = corner[:-1, :] * (u_pad[1:-1, :] - u_pad[:-2, :]) / dy     # (ny, nx+1)

    out[:, 1:-1] += (flux_n[:, 1:-1] - flux_s[:, 1:-1]) / dy
    return out


def _diff_v(v: np.ndarray, nu_cell: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Conservative variable-viscosity diffusion on v-momentum (ny+1, nx)."""
    ny_p1, nx = v.shape
    ny = ny_p1 - 1
    out = np.zeros_like(v)

    v_pad = np.zeros((ny_p1, nx + 2), dtype=v.dtype)
    v_pad[:, 1:-1] = v
    v_pad[:, 0]  = -v[:, 0]
    v_pad[:, -1] = -v[:, -1]

    # y-direction: d/dy(nu dv/dy). CV for v at (j, i) spans rows (j-1, j).
    flux_yc = nu_cell[:, :] * (v[1:, :] - v[:-1, :]) / dy  # (ny, nx)
    out[1:-1, :] = (flux_yc[1:, :] - flux_yc[:-1, :]) / dy  # (ny-1, nx)

    # x-direction: d/dx(nu dv/dx). Corner nu (shared with u-diff).
    nu_pad = np.zeros((ny + 2, nx + 2), dtype=nu_cell.dtype)
    nu_pad[1:-1, 1:-1] = nu_cell
    nu_pad[0,    1:-1] = nu_cell[0, :]
    nu_pad[-1,   1:-1] = nu_cell[-1, :]
    nu_pad[1:-1, 0]    = nu_cell[:, 0]
    nu_pad[1:-1, -1]   = nu_cell[:, -1]
    nu_pad[0,    0]    = nu_cell[0, 0]
    nu_pad[0,   -1]    = nu_cell[0, -1]
    nu_pad[-1,   0]    = nu_cell[-1, 0]
    nu_pad[-1,  -1]    = nu_cell[-1, -1]
    corner = 0.25 * (nu_pad[:-1, :-1] + nu_pad[1:, :-1] + nu_pad[:-1, 1:] + nu_pad[1:, 1:])  # (ny+1, nx+1)

    # East flux at v-CV (j, i), i in 1..nx-2: corner[j, i+1] * (v[j, i+1] - v[j, i])/dx
    flux_e = corner[:, 1:] * (v_pad[:, 2:] - v_pad[:, 1:-1]) / dx   # (ny+1, nx+1)  along x
    flux_w = corner[:, :-1] * (v_pad[:, 1:-1] - v_pad[:, :-2]) / dx  # (ny+1, nx+1)
    # Choose interior columns i=0..nx-1 → flux_e[:, 0..nx-1], flux_w[:, 0..nx-1]
    out[1:-1, :] += (flux_e[1:-1, :nx] - flux_w[1:-1, :nx]) / dx
    return out


def solve(
    config: Config,
    geom: Geometry,
    log: Callable[[str], None] = print,
) -> FlowField:
    nx, ny, dx, dy = geom.nx, geom.ny, geom.dx, geom.dy
    rho = config.fluid.rho_kg_m3
    nu = config.fluid.nu_m2_s
    v_in = config.bc.inlet_velocity_m_s
    v_fan = config.fan_velocity_m_s
    penalty = config.solver.brinkman_penalty_per_s
    alpha_p = config.solver.pressure_relax

    log(f"Building pressure solver: {int(geom.flow_mask.sum())} unknowns")
    lu, cell_idx, n_p, pinned = _build_pressure_solver(geom)

    flow = geom.flow_mask
    fan = geom.is_fan

    # Face pin masks / targets
    u_pin = np.zeros((ny, nx + 1), dtype=bool)
    u_pin[:, 1:-1] = ~(flow[:, :-1] & flow[:, 1:])
    u_pin[:, 0] = True
    u_pin[:, -1] = True
    u_pin[:, :-1] |= fan
    u_pin[:, 1:]  |= fan
    u_target = np.zeros((ny, nx + 1), dtype=float)

    v_pin = np.zeros((ny + 1, nx), dtype=bool)
    v_pin[1:-1, :] = ~(flow[:-1, :] & flow[1:, :])
    v_pin[0, :] = True
    v_pin[-1, :] = True
    v_pin[-1, geom.outlet_top_mask] = False          # outlet is zero-gradient, not pinned
    v_pin[geom.j_inlet, geom.inlet_face_mask] = True
    v_pin[:-1, :][fan] = True
    v_pin[1:,  :][fan] = True

    v_target = np.zeros((ny + 1, nx), dtype=float)
    v_target[geom.j_inlet, geom.inlet_face_mask] = v_in
    v_target[:-1, :][fan] = v_fan
    v_target[1:,  :][fan] = v_fan

    u = np.zeros((ny, nx + 1), dtype=float)
    v = np.zeros((ny + 1, nx), dtype=float)
    p = np.zeros((ny, nx), dtype=float)
    u[u_pin] = u_target[u_pin]
    v[v_pin] = v_target[v_pin]

    u_ref = max(v_in, v_fan)
    cfl = config.solver.cfl
    alpha_u = config.solver.momentum_relax

    # Interior-fluid mask: flow cells whose 4 faces are all free (none pinned).
    # These cells carry the "true" divergence residual; cells adjacent to
    # Brinkman pins absorb immersed-boundary correction errors.
    u_free = ~u_pin
    v_free = ~v_pin
    deep_interior = flow & np.roll(flow, -1, axis=1) & np.roll(flow, 1, axis=1) \
                         & np.roll(flow, -1, axis=0) & np.roll(flow, 1, axis=0)
    deep_interior[:, 0] = deep_interior[:, -1] = False
    deep_interior[0, :] = deep_interior[-1, :] = False
    # Also require all surrounding faces are unpinned (fluid-fluid)
    deep_interior &= u_free[:, :-1] & u_free[:, 1:]
    deep_interior &= v_free[:-1, :] & v_free[1:, :]

    div_scale = u_ref / min(dx, dy)
    div_target = max(config.solver.tolerance * div_scale, 1e-8)

    div_res = np.inf
    mom_res = np.inf
    div_history: list[float] = []
    mom_history: list[float] = []
    for it in range(1, config.solver.max_iterations + 1):
        nu_t_cell = (eddy_viscosity_cell(u, v, dx, dy, config.turbulence.mixing_length_m)
                     if config.turbulence.model == "mixing_length"
                     else np.zeros((ny, nx)))
        nu_t_cell = np.minimum(nu_t_cell, config.turbulence.nu_t_max_ratio * nu)
        nu_eff_cell = nu + nu_t_cell

        umax = max(np.abs(u).max(), np.abs(v).max(), u_ref)
        dt_conv = cfl * min(dx, dy) / umax
        dt_visc = 0.25 * min(dx, dy) ** 2 / max(nu_eff_cell.max(), 1e-12)
        dt = min(dt_conv, dt_visc)

        conv_u = _convect_u(u, v, dx, dy)
        conv_v = _convect_v(u, v, dx, dy)
        diff_u = _diff_u(u, nu_eff_cell, dx, dy)
        diff_v = _diff_v(v, nu_eff_cell, dx, dy)

        gradp_u = np.zeros((ny, nx + 1))
        gradp_u[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dx
        gradp_v = np.zeros((ny + 1, nx))
        gradp_v[1:-1, :] = (p[1:, :] - p[:-1, :]) / dy

        rhs_u = -conv_u + diff_u - gradp_u / rho
        rhs_v = -conv_v + diff_v - gradp_v / rho

        beta_u = np.where(u_pin, penalty, 0.0)
        beta_v = np.where(v_pin, penalty, 0.0)

        # Implicit Brinkman predictor
        u_star = (u + dt * rhs_u + dt * beta_u * u_target) / (1.0 + dt * beta_u)
        v_star = (v + dt * rhs_v + dt * beta_v * v_target) / (1.0 + dt * beta_v)

        # SIMPLE-style under-relaxation on the predictor (before projection).
        # Post-projection blending would destroy divergence-freeness.
        u_star = u + alpha_u * (u_star - u)
        v_star = v + alpha_u * (v_star - v)

        # Re-pin after predictor to anchor immersed-boundary cells exactly.
        u_star[u_pin] = u_target[u_pin]
        v_star[v_pin] = v_target[v_pin]
        v_star[-1, geom.outlet_top_mask] = v_star[-2, geom.outlet_top_mask]

        # Pressure correction: ∇²dp = (ρ/dt) ∇·u* on flow cells.
        div = (u_star[:, 1:] - u_star[:, :-1]) / dx + (v_star[1:, :] - v_star[:-1, :]) / dy
        rhs_p = (rho / dt) * div[flow]
        rhs_p[pinned] = 0.0
        dp_vec = lu.solve(rhs_p)
        dp = np.zeros((ny, nx))
        dp.ravel()[np.flatnonzero(flow.ravel())] = dp_vec

        p = p + alpha_p * dp

        u_new = u_star.copy()
        u_new[:, 1:-1] -= (dt / rho) * (dp[:, 1:] - dp[:, :-1]) / dx
        v_new = v_star.copy()
        v_new[1:-1, :] -= (dt / rho) * (dp[1:, :] - dp[:-1, :]) / dy

        # Re-pin after correction so that immersed-boundary targets (walls,
        # solids, fan faces, inlet line) stay exact. This introduces a small
        # divergence at those faces but is required for stability; all
        # *interior* flow cells remain divergence-free.
        u_new[u_pin] = u_target[u_pin]
        v_new[v_pin] = v_target[v_pin]
        v_new[-1, geom.outlet_top_mask] = v_new[-2, geom.outlet_top_mask]

        du_max = np.max(np.abs(u_new - u))
        dv_max = np.max(np.abs(v_new - v))
        mom_res = max(du_max, dv_max)

        if not np.isfinite(mom_res):
            log(f"  DIVERGED at iter {it}: non-finite momentum update")
            break

        u = u_new
        v = v_new

        div_post = (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy
        div_all = np.abs(div_post[flow]).max()
        div_interior = np.abs(div_post[deep_interior]).max() if deep_interior.any() else 0.0
        div_res = div_interior
        div_history.append(float(div_interior))
        mom_history.append(float(mom_res))

        if it % config.solver.log_every == 0 or it == 1:
            log(f"  iter {it:5d}  div_int={div_interior:.2e}  div_all={div_all:.2e}"
                f"  dU={mom_res:.2e}  Vmax={umax:.2f}  ν_t/ν max={(nu_t_cell.max()/nu):.1f}")

        converged = (div_res < div_target) and (mom_res < config.solver.tolerance * u_ref)
        if converged and it > 50:
            log(f"  Converged at iter {it}: div_int={div_interior:.2e}, dU={mom_res:.2e}")
            break
    else:
        log(f"  Reached max_iterations={config.solver.max_iterations}, "
            f"div_int={div_interior:.2e}, div_all={div_all:.2e}, dU={mom_res:.2e}")

    u_cell = 0.5 * (u[:, :-1] + u[:, 1:])
    v_cell = 0.5 * (v[:-1, :] + v[1:, :])
    speed_cell = np.sqrt(u_cell ** 2 + v_cell ** 2)

    dvdx_cell = np.zeros((ny, nx))
    dudy_cell = np.zeros((ny, nx))
    dvdx_cell[:, 1:-1] = (v_cell[:, 2:] - v_cell[:, :-2]) / (2 * dx)
    dudy_cell[1:-1, :] = (u_cell[2:, :] - u_cell[:-2, :]) / (2 * dy)
    vort = dvdx_cell - dudy_cell
    vort[geom.is_solid_flow & ~geom.is_fan] = 0.0

    return FlowField(
        u=u, v=v, p=p,
        u_cell=u_cell, v_cell=v_cell, speed_cell=speed_cell, vorticity_cell=vort,
        iterations=it,
        final_residual=div_res,
        final_momentum_res=mom_res,
        converged=(div_res < div_target) and (mom_res < config.solver.tolerance * u_ref),
        div_history=div_history,
        mom_history=mom_history,
    )
