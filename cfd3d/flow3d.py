"""3D incompressible Navier–Stokes on a staggered MAC grid.

Algorithm mirrors the 2D solver:

    1. Momentum predictor with explicit first-order upwind advection,
       constant-viscosity diffusion, previous-step pressure gradient,
       and implicit Brinkman drag for solid / fan / inlet faces.
    2. Pressure-correction Poisson on flow cells with Neumann BCs at
       walls / solids / inlet and Dirichlet dp=0 at outlet faces.
    3. Velocity correction via grad(dp); the velocity field is
       projected to divergence-free on the interior.
    4. Re-pin Brinkman faces (immersed-boundary anchor).

The 3D pressure Poisson is solved matrix-free with CG + Jacobi
preconditioning (scipy.sparse.linalg.cg + LinearOperator). Direct
factorization is avoided because fill-in in 3D is prohibitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from .config3d import Config3D
from .geometry3d import Geometry3D


@dataclass
class FlowField3D:
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    p: np.ndarray
    u_cell: np.ndarray
    v_cell: np.ndarray
    w_cell: np.ndarray
    speed_cell: np.ndarray
    iterations: int
    final_div: float
    final_mom_res: float
    converged: bool
    div_history: list
    mom_history: list


def _upwind_first(phi: np.ndarray, vel: np.ndarray, h: float, axis: int) -> np.ndarray:
    """First-order upwind derivative ∂phi/∂x_axis using velocity sign.

    `phi` and `vel` must have the same shape. The derivative at the
    interior points of the chosen axis uses backward difference when
    vel>0 and forward when vel<0. Boundary slabs are set to zero.
    """
    slicer = [slice(None)] * phi.ndim
    out = np.zeros_like(phi)
    # backward: phi[i] - phi[i-1]
    s_mid = slicer.copy();  s_mid[axis] = slice(1, -1)
    s_back = slicer.copy(); s_back[axis] = slice(0, -2)
    s_fwd = slicer.copy();  s_fwd[axis] = slice(2, None)
    d_back = (phi[tuple(s_mid)] - phi[tuple(s_back)]) / h
    d_fwd  = (phi[tuple(s_fwd)] - phi[tuple(s_mid)]) / h
    v_mid = vel[tuple(s_mid)]
    out[tuple(s_mid)] = np.where(v_mid > 0, d_back, d_fwd)
    return out


def _central_second(phi: np.ndarray, phi_ghost_lo: np.ndarray,
                    phi_ghost_hi: np.ndarray, h: float, axis: int) -> np.ndarray:
    """Central second difference along `axis` with Dirichlet ghost slabs."""
    pad_shape = list(phi.shape)
    pad_shape[axis] += 2
    padded = np.zeros(pad_shape, dtype=phi.dtype)
    inner = [slice(None)] * phi.ndim
    inner[axis] = slice(1, -1)
    padded[tuple(inner)] = phi
    lo = [slice(None)] * phi.ndim
    lo[axis] = slice(0, 1)
    hi = [slice(None)] * phi.ndim
    hi[axis] = slice(-1, None)
    padded[tuple(lo)] = phi_ghost_lo
    padded[tuple(hi)] = phi_ghost_hi
    s0 = [slice(None)] * phi.ndim; s0[axis] = slice(0, -2)
    s1 = [slice(None)] * phi.ndim; s1[axis] = slice(1, -1)
    s2 = [slice(None)] * phi.ndim; s2[axis] = slice(2, None)
    return (padded[tuple(s2)] - 2 * padded[tuple(s1)] + padded[tuple(s0)]) / (h * h)


def _convect_u(u: np.ndarray, v: np.ndarray, w: np.ndarray,
               dx: float, dy: float, dz: float) -> np.ndarray:
    """Advective term (u·∇)u on u-faces, shape (nz, ny, nx+1)."""
    # Interpolate v, w to u-face positions (k, j, i+0.5) for i in 1..nx-1.
    # u-face at (k,j,i) with i interior lies between cells (k,j,i-1) and (k,j,i).
    # v at u-face: average of v[k,j,i-1], v[k,j+1,i-1], v[k,j,i], v[k,j+1,i] → (nz, ny, nx-1) for i in [1,nx-1]
    v_left = v[:, :, :-1]; v_right = v[:, :, 1:]                         # (nz, ny+1, nx-1)
    v_at_u = 0.25 * (v_left[:, :-1] + v_left[:, 1:] + v_right[:, :-1] + v_right[:, 1:])  # (nz, ny, nx-1)
    w_left = w[:, :, :-1]; w_right = w[:, :, 1:]                         # (nz+1, ny, nx-1)
    w_at_u = 0.25 * (w_left[:-1, :] + w_left[1:, :] + w_right[:-1, :] + w_right[1:, :])  # (nz, ny, nx-1)

    u_int = u[:, :, 1:-1]

    # ∂u/∂x upwind on u along axis=2.
    du_dx_b = (u[:, :, 1:-1] - u[:, :, :-2]) / dx
    du_dx_f = (u[:, :, 2:]   - u[:, :, 1:-1]) / dx
    du_dx = np.where(u_int > 0, du_dx_b, du_dx_f)

    # ∂u/∂y with no-slip ghost rows in y.
    u_pad_y = np.zeros((u.shape[0], u.shape[1] + 2, u.shape[2]), dtype=u.dtype)
    u_pad_y[:, 1:-1, :] = u
    u_pad_y[:, 0, :]  = -u[:, 0, :]
    u_pad_y[:, -1, :] = -u[:, -1, :]
    du_dy_b = (u_pad_y[:, 1:-1, 1:-1] - u_pad_y[:, :-2, 1:-1]) / dy
    du_dy_f = (u_pad_y[:, 2:,   1:-1] - u_pad_y[:, 1:-1, 1:-1]) / dy
    du_dy = np.where(v_at_u > 0, du_dy_b, du_dy_f)

    # ∂u/∂z with no-slip ghost slabs in z.
    u_pad_z = np.zeros((u.shape[0] + 2, u.shape[1], u.shape[2]), dtype=u.dtype)
    u_pad_z[1:-1, :, :] = u
    u_pad_z[0, :, :]  = -u[0, :, :]
    u_pad_z[-1, :, :] = -u[-1, :, :]
    du_dz_b = (u_pad_z[1:-1, :, 1:-1] - u_pad_z[:-2, :, 1:-1]) / dz
    du_dz_f = (u_pad_z[2:,   :, 1:-1] - u_pad_z[1:-1, :, 1:-1]) / dz
    du_dz = np.where(w_at_u > 0, du_dz_b, du_dz_f)

    conv = np.zeros_like(u)
    conv[:, :, 1:-1] = u_int * du_dx + v_at_u * du_dy + w_at_u * du_dz
    return conv


def _convect_v(u: np.ndarray, v: np.ndarray, w: np.ndarray,
               dx: float, dy: float, dz: float) -> np.ndarray:
    # u at v-face (k,j,i): average u[k,j-1,i:i+2]_nonfaced... simpler via cell-pairs.
    u_top = u[:, :-1, :]; u_bot = u[:, 1:, :]                        # (nz, ny, nx+1)
    u_at_v = 0.25 * (u_top[:, :, :-1] + u_top[:, :, 1:] + u_bot[:, :, :-1] + u_bot[:, :, 1:])  # (nz, ny, nx)
    # Match to interior v-faces j=1..ny-1: v shape (nz, ny+1, nx). v interior: v[:, 1:-1, :].
    # u_at_v computed above has shape (nz, ny, nx) — this is u averaged to horizontal midplane
    # between cell rows j and j+1 (i.e., at v-face j+1). Convert to v-face indexing:
    # v-face at row j has neighbors cells (j-1, j). u_at_v_row[j] = from cells j-1, j.
    # u_at_v_row[j] = 0.25*(u[k,j-1,i]+u[k,j-1,i+1]+u[k,j,i]+u[k,j,i+1]) corresponds to u_top/u_bot
    # where u_top=u[j-1], u_bot=u[j]. In the formula above we used u_top = u[:, :-1, :] = rows 0..ny-2
    # and u_bot = u[:, 1:, :] = rows 1..ny-1. That's u_top_row = j, u_bot_row = j+1 for j=0..ny-2,
    # giving u at "v-face between rows j and j+1", i.e., v-face index j+1 (interior 1..ny-1). So
    # u_at_v corresponds to v_interior = v[:, 1:-1, :] after indexing 0..ny-2 maps to v_face 1..ny-1.
    # → u_at_v has shape (nz, ny, nx) matching v[:, 1:-1, :] with row offset 0.
    w_top = w[:, :-1, :]; w_bot = w[:, 1:, :]                        # (nz+1, ny, nx)
    w_at_v = 0.25 * (w_top[:-1, :, :] + w_top[1:, :, :] + w_bot[:-1, :, :] + w_bot[1:, :, :])  # (nz, ny, nx)

    v_int = v[:, 1:-1, :]

    dv_dy_b = (v[:, 1:-1, :] - v[:, :-2, :]) / dy
    dv_dy_f = (v[:, 2:,   :] - v[:, 1:-1, :]) / dy
    dv_dy = np.where(v_int > 0, dv_dy_b, dv_dy_f)

    v_pad_x = np.zeros((v.shape[0], v.shape[1], v.shape[2] + 2), dtype=v.dtype)
    v_pad_x[:, :, 1:-1] = v
    v_pad_x[:, :, 0]  = -v[:, :, 0]
    v_pad_x[:, :, -1] = -v[:, :, -1]
    dv_dx_b = (v_pad_x[:, 1:-1, 1:-1] - v_pad_x[:, 1:-1, :-2]) / dx
    dv_dx_f = (v_pad_x[:, 1:-1, 2:]   - v_pad_x[:, 1:-1, 1:-1]) / dx
    dv_dx = np.where(u_at_v > 0, dv_dx_b, dv_dx_f)

    v_pad_z = np.zeros((v.shape[0] + 2, v.shape[1], v.shape[2]), dtype=v.dtype)
    v_pad_z[1:-1, :, :] = v
    v_pad_z[0, :, :]  = -v[0, :, :]
    v_pad_z[-1, :, :] = -v[-1, :, :]
    dv_dz_b = (v_pad_z[1:-1, 1:-1, :] - v_pad_z[:-2, 1:-1, :]) / dz
    dv_dz_f = (v_pad_z[2:,   1:-1, :] - v_pad_z[1:-1, 1:-1, :]) / dz
    dv_dz = np.where(w_at_v > 0, dv_dz_b, dv_dz_f)

    conv = np.zeros_like(v)
    conv[:, 1:-1, :] = u_at_v * dv_dx + v_int * dv_dy + w_at_v * dv_dz
    return conv


def _convect_w(u: np.ndarray, v: np.ndarray, w: np.ndarray,
               dx: float, dy: float, dz: float) -> np.ndarray:
    u_below = u[:-1, :, :]; u_above = u[1:, :, :]                    # (nz, ny, nx+1)
    u_at_w = 0.25 * (u_below[:, :, :-1] + u_below[:, :, 1:] + u_above[:, :, :-1] + u_above[:, :, 1:])  # (nz, ny, nx)
    v_below = v[:-1, :, :]; v_above = v[1:, :, :]                    # (nz, ny+1, nx)
    v_at_w = 0.25 * (v_below[:, :-1, :] + v_below[:, 1:, :] + v_above[:, :-1, :] + v_above[:, 1:, :])  # (nz, ny, nx)

    w_int = w[1:-1, :, :]

    dw_dz_b = (w[1:-1, :, :] - w[:-2, :, :]) / dz
    dw_dz_f = (w[2:,   :, :] - w[1:-1, :, :]) / dz
    dw_dz = np.where(w_int > 0, dw_dz_b, dw_dz_f)

    w_pad_x = np.zeros((w.shape[0], w.shape[1], w.shape[2] + 2), dtype=w.dtype)
    w_pad_x[:, :, 1:-1] = w
    w_pad_x[:, :, 0]  = -w[:, :, 0]
    w_pad_x[:, :, -1] = -w[:, :, -1]
    dw_dx_b = (w_pad_x[1:-1, :, 1:-1] - w_pad_x[1:-1, :, :-2]) / dx
    dw_dx_f = (w_pad_x[1:-1, :, 2:]   - w_pad_x[1:-1, :, 1:-1]) / dx
    dw_dx = np.where(u_at_w > 0, dw_dx_b, dw_dx_f)

    w_pad_y = np.zeros((w.shape[0], w.shape[1] + 2, w.shape[2]), dtype=w.dtype)
    w_pad_y[:, 1:-1, :] = w
    w_pad_y[:, 0, :]  = -w[:, 0, :]
    w_pad_y[:, -1, :] = -w[:, -1, :]
    dw_dy_b = (w_pad_y[1:-1, 1:-1, :] - w_pad_y[1:-1, :-2, :]) / dy
    dw_dy_f = (w_pad_y[1:-1, 2:,   :] - w_pad_y[1:-1, 1:-1, :]) / dy
    dw_dy = np.where(v_at_w > 0, dw_dy_b, dw_dy_f)

    conv = np.zeros_like(w)
    conv[1:-1, :, :] = u_at_w * dw_dx + v_at_w * dw_dy + w_int * dw_dz
    return conv


def _scalar_laplacian(phi: np.ndarray, dx: float, dy: float, dz: float,
                       ghost_sign: int = -1) -> np.ndarray:
    """3D central Laplacian with antisymmetric (no-slip) ghost slabs."""
    zero_lo = ghost_sign * np.take(phi, [0], axis=0)
    zero_hi = ghost_sign * np.take(phi, [-1], axis=0)
    lap = _central_second(phi, zero_lo, zero_hi, dz, axis=0)
    zero_lo = ghost_sign * np.take(phi, [0], axis=1)
    zero_hi = ghost_sign * np.take(phi, [-1], axis=1)
    lap += _central_second(phi, zero_lo, zero_hi, dy, axis=1)
    zero_lo = ghost_sign * np.take(phi, [0], axis=2)
    zero_hi = ghost_sign * np.take(phi, [-1], axis=2)
    lap += _central_second(phi, zero_lo, zero_hi, dx, axis=2)
    return lap


def _build_pressure_solver(geom: Geometry3D, log: Callable[[str], None] = print):
    """Assemble and factorize the 3D pressure-correction Poisson matrix.

    Identical approach to the 2D solver: enumerate flow cells, build
    a sparse matrix with the 7-point stencil, factorize once with splu.
    At ~60k–90k flow-cell unknowns the fill-in is manageable (~50 MB).
    """
    nx, ny, nz = geom.nx, geom.ny, geom.nz
    dx, dy, dz = geom.dx, geom.dy, geom.dz
    inv_dx2, inv_dy2, inv_dz2 = 1.0/(dx*dx), 1.0/(dy*dy), 1.0/(dz*dz)

    flow = geom.flow_mask
    cell_idx = -np.ones((nz, ny, nx), dtype=np.int64)
    flat = np.flatnonzero(flow.ravel())
    cell_idx.ravel()[flat] = np.arange(flat.size)
    n_unknowns = flat.size
    log(f"  pressure unknowns: {n_unknowns}")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diag_arr = np.zeros(n_unknowns, dtype=float)
    has_dirichlet = np.zeros(n_unknowns, dtype=bool)

    ks, js, is_ = np.where(flow)
    for idx_n, (k, j, i) in enumerate(zip(ks.tolist(), js.tolist(), is_.tolist())):
        # -x
        if i - 1 >= 0 and flow[k, j, i - 1]:
            rows.append(idx_n); cols.append(int(cell_idx[k, j, i-1])); data.append(inv_dx2)
            diag_arr[idx_n] -= inv_dx2
        # +x
        if i + 1 < nx and flow[k, j, i + 1]:
            rows.append(idx_n); cols.append(int(cell_idx[k, j, i+1])); data.append(inv_dx2)
            diag_arr[idx_n] -= inv_dx2
        # -y
        if j - 1 >= 0 and flow[k, j - 1, i]:
            rows.append(idx_n); cols.append(int(cell_idx[k, j-1, i])); data.append(inv_dy2)
            diag_arr[idx_n] -= inv_dy2
        # +y
        if j + 1 < ny and flow[k, j + 1, i]:
            rows.append(idx_n); cols.append(int(cell_idx[k, j+1, i])); data.append(inv_dy2)
            diag_arr[idx_n] -= inv_dy2
        elif j == ny - 1 and geom.outlet_rear_mask[k, i]:
            diag_arr[idx_n] -= inv_dy2
            has_dirichlet[idx_n] = True
        # -z
        if k - 1 >= 0 and flow[k - 1, j, i]:
            rows.append(idx_n); cols.append(int(cell_idx[k-1, j, i])); data.append(inv_dz2)
            diag_arr[idx_n] -= inv_dz2
        # +z
        if k + 1 < nz and flow[k + 1, j, i]:
            rows.append(idx_n); cols.append(int(cell_idx[k+1, j, i])); data.append(inv_dz2)
            diag_arr[idx_n] -= inv_dz2

    # Pin isolated components (no Dirichlet anchor)
    from scipy.sparse import csr_matrix as csr
    from scipy.sparse.csgraph import connected_components
    if n_unknowns:
        adj = csr(
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


def _build_pin_fields(geom: Geometry3D, v_in: float, v_fan: float):
    """Build face-level pin masks and targets for Brinkman penalization."""
    nx, ny, nz = geom.nx, geom.ny, geom.nz
    flow = geom.flow_mask
    fan = geom.is_fan

    # u-face pin: any face adjacent to non-flow cell, plus x=0 / x=nx walls, plus fan cell faces
    u_pin = np.zeros((nz, ny, nx + 1), dtype=bool)
    u_pin[:, :, 1:-1] = ~(flow[:, :, :-1] & flow[:, :, 1:])
    u_pin[:, :, 0] = True
    u_pin[:, :, -1] = True
    u_pin[:, :, :-1] |= fan
    u_pin[:, :,  1:] |= fan

    v_pin = np.zeros((nz, ny + 1, nx), dtype=bool)
    v_pin[:, 1:-1, :] = ~(flow[:, :-1, :] & flow[:, 1:, :])
    v_pin[:, 0, :] = True
    v_pin[:, -1, :] = True
    v_pin[:, :-1, :] |= fan
    v_pin[:,  1:, :] |= fan

    w_pin = np.zeros((nz + 1, ny, nx), dtype=bool)
    w_pin[1:-1, :, :] = ~(flow[:-1, :, :] & flow[1:, :, :])
    w_pin[0,  :, :] = True
    w_pin[-1, :, :] = True
    w_pin[:-1, :, :] |= fan
    w_pin[1:,  :, :] |= fan

    u_target = np.zeros_like(u_pin, dtype=float)
    v_target = np.zeros_like(v_pin, dtype=float)
    w_target = np.zeros_like(w_pin, dtype=float)

    # Inlet: bottom face (w on z=0) inside the inlet mask. w-faces index 0 is the
    # boundary face of cells in the bottom slab.
    w_target[0, :, :][geom.inlet_bottom_mask] = v_in
    # Leave w_pin[0] already True from the wall pin above (we overwrite the target).

    # Outlets: rear y-face (ny index) inside outlet_rear_mask.
    # These faces are NOT pinned — zero-gradient outflow.
    v_pin[:, -1, :][geom.outlet_rear_mask] = False

    # Fan faces: direction depends on fan_axis. Pin both inflow and outflow faces.
    # fan_axis codes: 1=+y, 2=-y, 3=+x, 4=-x. All fans in MacBook blow toward +y
    # (rear hinge) in our simplified model.
    fan_axis = geom.fan_axis
    fan_plus_y  = fan & (fan_axis == 1)
    fan_minus_y = fan & (fan_axis == 2)
    fan_plus_x  = fan & (fan_axis == 3)
    fan_minus_x = fan & (fan_axis == 4)

    # +y fan: pin +y and -y faces to +v_fan (axial flow through the fan block)
    v_pin[:, :-1, :][fan_plus_y] = True
    v_pin[:,  1:, :][fan_plus_y] = True
    v_target[:, :-1, :][fan_plus_y] = v_fan
    v_target[:,  1:, :][fan_plus_y] = v_fan
    # -y fan
    v_pin[:, :-1, :][fan_minus_y] = True
    v_pin[:,  1:, :][fan_minus_y] = True
    v_target[:, :-1, :][fan_minus_y] = -v_fan
    v_target[:,  1:, :][fan_minus_y] = -v_fan
    # ±x fans
    u_pin[:, :, :-1][fan_plus_x] = True
    u_pin[:, :,  1:][fan_plus_x] = True
    u_target[:, :, :-1][fan_plus_x] = v_fan
    u_target[:, :,  1:][fan_plus_x] = v_fan
    u_pin[:, :, :-1][fan_minus_x] = True
    u_pin[:, :,  1:][fan_minus_x] = True
    u_target[:, :, :-1][fan_minus_x] = -v_fan
    u_target[:, :,  1:][fan_minus_x] = -v_fan

    return (u_pin, u_target), (v_pin, v_target), (w_pin, w_target)


def solve(config: Config3D, geom: Geometry3D,
          log: Callable[[str], None] = print) -> FlowField3D:
    nx, ny, nz = geom.nx, geom.ny, geom.nz
    dx, dy, dz = geom.dx, geom.dy, geom.dz
    rho = config.fluid.rho_kg_m3
    nu = config.fluid.nu_m2_s
    v_in = config.bc.inlet.velocity_m_s
    v_fan = config.fan_velocity_m_s
    penalty = config.solver.brinkman_penalty_per_s
    alpha_p = config.solver.pressure_relax
    alpha_u = config.solver.momentum_relax

    flow = geom.flow_mask

    log(f"Building pressure solver ...")
    lu, cell_idx, n_p, pinned = _build_pressure_solver(geom, log=log)

    (u_pin, u_target), (v_pin, v_target), (w_pin, w_target) = \
        _build_pin_fields(geom, v_in, v_fan)

    u = np.zeros((nz, ny, nx + 1), dtype=float)
    v = np.zeros((nz, ny + 1, nx), dtype=float)
    w = np.zeros((nz + 1, ny, nx), dtype=float)
    p = np.zeros((nz, ny, nx), dtype=float)
    u[u_pin] = u_target[u_pin]
    v[v_pin] = v_target[v_pin]
    w[w_pin] = w_target[w_pin]

    u_ref = max(v_in, v_fan, 1e-6)
    cfl = config.solver.cfl
    h_min = min(dx, dy, dz)

    deep = (
        flow
        & np.roll(flow, -1, axis=2) & np.roll(flow, 1, axis=2)
        & np.roll(flow, -1, axis=1) & np.roll(flow, 1, axis=1)
        & np.roll(flow, -1, axis=0) & np.roll(flow, 1, axis=0)
    )
    deep[:, :, 0] = deep[:, :, -1] = False
    deep[:, 0, :] = deep[:, -1, :] = False
    deep[0, :, :] = deep[-1, :, :] = False

    div_target = max(config.solver.tolerance * u_ref / h_min, 1e-8)

    beta_u = np.where(u_pin, penalty, 0.0)
    beta_v = np.where(v_pin, penalty, 0.0)
    beta_w = np.where(w_pin, penalty, 0.0)

    div_interior = np.inf
    mom_res = np.inf
    div_history: list[float] = []
    mom_history: list[float] = []

    for it in range(1, config.solver.max_iterations + 1):
        umax_abs = max(np.abs(u).max(), np.abs(v).max(), np.abs(w).max(), u_ref)
        dt_conv = cfl * h_min / umax_abs
        dt_visc = 0.25 * h_min ** 2 / nu
        dt = min(dt_conv, dt_visc)

        conv_u = _convect_u(u, v, w, dx, dy, dz)
        conv_v = _convect_v(u, v, w, dx, dy, dz)
        conv_w = _convect_w(u, v, w, dx, dy, dz)

        lap_u = _scalar_laplacian(u, dx, dy, dz)
        lap_v = _scalar_laplacian(v, dx, dy, dz)
        lap_w = _scalar_laplacian(w, dx, dy, dz)

        gradp_u = np.zeros_like(u)
        gradp_u[:, :, 1:-1] = (p[:, :, 1:] - p[:, :, :-1]) / dx
        gradp_v = np.zeros_like(v)
        gradp_v[:, 1:-1, :] = (p[:, 1:, :] - p[:, :-1, :]) / dy
        gradp_w = np.zeros_like(w)
        gradp_w[1:-1, :, :] = (p[1:, :, :] - p[:-1, :, :]) / dz

        rhs_u = -conv_u + nu * lap_u - gradp_u / rho
        rhs_v = -conv_v + nu * lap_v - gradp_v / rho
        rhs_w = -conv_w + nu * lap_w - gradp_w / rho

        u_star = (u + dt * rhs_u + dt * beta_u * u_target) / (1.0 + dt * beta_u)
        v_star = (v + dt * rhs_v + dt * beta_v * v_target) / (1.0 + dt * beta_v)
        w_star = (w + dt * rhs_w + dt * beta_w * w_target) / (1.0 + dt * beta_w)

        u_star = u + alpha_u * (u_star - u)
        v_star = v + alpha_u * (v_star - v)
        w_star = w + alpha_u * (w_star - w)

        u_star[u_pin] = u_target[u_pin]
        v_star[v_pin] = v_target[v_pin]
        w_star[w_pin] = w_target[w_pin]
        v_star[:, -1, :][geom.outlet_rear_mask] = v_star[:, -2, :][geom.outlet_rear_mask]

        # Pressure correction via direct solve (splu)
        div = (
            (u_star[:, :, 1:] - u_star[:, :, :-1]) / dx
            + (v_star[:, 1:, :] - v_star[:, :-1, :]) / dy
            + (w_star[1:, :, :] - w_star[:-1, :, :]) / dz
        )
        rhs_p = (rho / dt) * div[flow]
        rhs_p[pinned] = 0.0
        dp_vec = lu.solve(rhs_p)
        dp = np.zeros((nz, ny, nx))
        dp.ravel()[np.flatnonzero(flow.ravel())] = dp_vec

        p = p + alpha_p * dp

        u_new = u_star.copy()
        u_new[:, :, 1:-1] -= (dt / rho) * (dp[:, :, 1:] - dp[:, :, :-1]) / dx
        v_new = v_star.copy()
        v_new[:, 1:-1, :] -= (dt / rho) * (dp[:, 1:, :] - dp[:, :-1, :]) / dy
        w_new = w_star.copy()
        w_new[1:-1, :, :] -= (dt / rho) * (dp[1:, :, :] - dp[:-1, :, :]) / dz

        u_new[u_pin] = u_target[u_pin]
        v_new[v_pin] = v_target[v_pin]
        w_new[w_pin] = w_target[w_pin]
        v_new[:, -1, :][geom.outlet_rear_mask] = v_new[:, -2, :][geom.outlet_rear_mask]

        du = np.max(np.abs(u_new - u))
        dvv = np.max(np.abs(v_new - v))
        dww = np.max(np.abs(w_new - w))
        mom_res = max(du, dvv, dww)

        if not np.isfinite(mom_res):
            log(f"  DIVERGED at iter {it}: non-finite update")
            break

        u, v, w = u_new, v_new, w_new

        div_post = (
            (u[:, :, 1:] - u[:, :, :-1]) / dx
            + (v[:, 1:, :] - v[:, :-1, :]) / dy
            + (w[1:, :, :] - w[:-1, :, :]) / dz
        )
        div_interior = np.abs(div_post[deep]).max() if deep.any() else 0.0
        div_all = np.abs(div_post[flow]).max()

        div_history.append(float(div_interior))
        mom_history.append(float(mom_res))

        if it % config.solver.log_every == 0 or it == 1:
            log(f"  iter {it:4d}  div_int={div_interior:.2e}  div_all={div_all:.2e}"
                f"  dU={mom_res:.2e}  dt={dt:.1e}  Vmax={umax_abs:.2f}")

        converged = (div_interior < div_target) and (mom_res < config.solver.tolerance * u_ref)
        if converged and it > 50:
            log(f"  Converged at iter {it}")
            break
    else:
        log(f"  Reached max_iterations={config.solver.max_iterations}")

    u_cell = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    v_cell = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    w_cell = 0.5 * (w[:-1, :, :] + w[1:, :, :])
    speed = np.sqrt(u_cell ** 2 + v_cell ** 2 + w_cell ** 2)

    return FlowField3D(
        u=u, v=v, w=w, p=p,
        u_cell=u_cell, v_cell=v_cell, w_cell=w_cell, speed_cell=speed,
        iterations=it,
        final_div=float(div_interior),
        final_mom_res=float(mom_res),
        converged=bool((div_interior < div_target) and (mom_res < config.solver.tolerance * u_ref)),
        div_history=div_history,
        mom_history=mom_history,
    )
