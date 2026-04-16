"""
2D Natural Convection CFD — MacBook Cross-Section (Lid Closed)
================================================================
Conjugate heat transfer with stream function-vorticity formulation.

Key numerical techniques:
  - DIRECT sparse solve for temperature (handles 5000:1 k contrast)
  - Harmonic-mean face conductivities at material interfaces
  - Direct sparse LU for stream function Poisson (fluid-only)
  - Upwind differencing for convection stability
  - Iterative relaxation for vorticity transport
  - Thom's formula at enclosure walls and internal solid surfaces
"""

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import splu, spsolve
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ================================================================
# 1. PHYSICAL CONSTANTS
# ================================================================

L, H = 0.300, 0.015
T_BOT, T_TOP = 28.0, 32.0
DT_WALL = T_TOP - T_BOT

RHO, MU = 1.127, 1.91e-5
K_AIR, CP = 0.02756, 1007.0
BETA, GRAV = 3.19e-3, 9.81

NU = MU / RHO
ALPHA_AIR = K_AIR / (RHO * CP)

# ================================================================
# 2. COMPONENT DEFINITIONS
# ================================================================

COMPONENTS = [
    ("M-series SoC",   120, 145,   2,  7,  8.0,   150.0, 2330,  700),
    ("LPDDR RAM",      148, 165,   2,  5,  1.5,     1.5, 1800,  900),
    ("SSD Controller",  80,  95,   2,  5,  1.0,     1.0, 1800,  900),
    ("Battery (L)",     10,  70,   2, 12,  0.5,     3.0, 2500, 1000),
    ("Battery (R)",    200, 270,   2, 12,  0.5,     3.0, 2500, 1000),
    ("USB-C (L)",        2,  10,   2,  6,  0.3,    20.0, 2000,  800),
    ("USB-C (R)",      290, 298,   2,  6,  0.3,    20.0, 2000,  800),
    ("Vapor Chamber",  110, 180,   7,  9,  0.0, 10000.0, 8900,  385),
]

# ================================================================
# 3. GRID
# ================================================================

NX, NY = 200, 50
dx, dy = L / (NX - 1), H / (NY - 1)
dx2, dy2 = dx * dx, dy * dy
x = np.linspace(0, L, NX)
y = np.linspace(0, H, NY)
X, Y = np.meshgrid(x, y)

# ================================================================
# 4. SOLID MASKING
# ================================================================

is_solid = np.zeros((NY, NX), dtype=bool)
comp_id = -np.ones((NY, NX), dtype=int)

for ci, (_, x0, x1, y0, y1, *_) in enumerate(COMPONENTS):
    mask = ((X >= x0 * 1e-3) & (X <= x1 * 1e-3) &
            (Y >= y0 * 1e-3) & (Y <= y1 * 1e-3))
    is_solid |= mask
    comp_id[mask] = ci

is_fluid = ~is_solid
fluid_int = np.zeros((NY, NX), dtype=bool)
fluid_int[1:-1, 1:-1] = is_fluid[1:-1, 1:-1]

# ================================================================
# 5. MATERIAL PROPERTIES AND FACE CONDUCTIVITIES
# ================================================================

k_arr = np.full((NY, NX), K_AIR)
for ci, (_, _, _, _, _, _, k_s, _, _) in enumerate(COMPONENTS):
    k_arr[comp_id == ci] = k_s

Q_vol = np.zeros((NY, NX))
for ci, (_, x0, x1, y0, y1, Q_w, *_) in enumerate(COMPONENTS):
    if Q_w > 0:
        area = (x1 - x0) * 1e-3 * (y1 - y0) * 1e-3
        Q_vol[comp_id == ci] = Q_w / area

# Harmonic-mean face conductivities
kx_face = np.zeros((NY, NX))
kx_face[:, :-1] = 2.0 * k_arr[:, :-1] * k_arr[:, 1:] / (k_arr[:, :-1] + k_arr[:, 1:])
ky_face = np.zeros((NY, NX))
ky_face[:-1, :] = 2.0 * k_arr[:-1, :] * k_arr[1:, :] / (k_arr[:-1, :] + k_arr[1:, :])

# Face diffusion conductances for interior cells
De = kx_face[1:-1, 1:-1].copy() / dx2;  De[:, -1] = 0.0
Dw = np.zeros((NY - 2, NX - 2));        Dw[:, 1:] = kx_face[1:-1, 1:-2] / dx2
Dn = ky_face[1:-1, 1:-1].copy() / dy2;  Dn[-1, :] = k_arr[NY - 2, 1:-1] / dy2
Ds = np.zeros((NY - 2, NX - 2))
Ds[1:, :] = ky_face[1:-2, 1:-1] / dy2
Ds[0, :] = k_arr[1, 1:-1] / dy2

Q_int = Q_vol[1:-1, 1:-1]

# ================================================================
# 6. TEMPERATURE MATRIX (direct sparse solve)
# ================================================================
# Build the sparse matrix for  div(k grad T) + Q = rho cp (u.grad T)
# using harmonic-mean face conductivities.  Dirichlet BCs at
# top/bottom folded into the RHS; Neumann at left/right via D=0.

print("Building temperature matrix ...")

n_x, n_y = NX - 2, NY - 2
n_T = n_x * n_y

jj_int, ii_int = np.mgrid[0:n_y, 0:n_x]
kk = (jj_int * n_x + ii_int)   # linear index

# Diffusion part of the matrix (constant)
rows_d, cols_d, vals_d = [], [], []

# Centre diagonal: -(De + Dw + Dn + Ds)
rows_d.append(kk.ravel())
cols_d.append(kk.ravel())
vals_d.append(-(De + Dw + Dn + Ds).ravel())

# East (i+1): only interior neighbours
m = (De > 0).ravel()
rows_d.append(kk.ravel()[m]);  cols_d.append((kk + 1).ravel()[m])
vals_d.append(De.ravel()[m])

# West (i-1)
m = (Dw > 0).ravel()
rows_d.append(kk.ravel()[m]);  cols_d.append((kk - 1).ravel()[m])
vals_d.append(Dw.ravel()[m])

# North (j+1): only if north neighbour is interior (not top wall)
m = (jj_int < n_y - 1).ravel()
rows_d.append(kk.ravel()[m]);  cols_d.append((kk + n_x).ravel()[m])
vals_d.append(Dn.ravel()[m])

# South (j-1): only if south neighbour is interior (not bottom wall)
m = (jj_int > 0).ravel()
rows_d.append(kk.ravel()[m]);  cols_d.append((kk - n_x).ravel()[m])
vals_d.append(Ds.ravel()[m])

r_d = np.concatenate(rows_d)
c_d = np.concatenate(cols_d)
v_d = np.concatenate(vals_d)
A_diff = coo_matrix((v_d, (r_d, c_d)), shape=(n_T, n_T)).tocsc()

# Constant part of the RHS: -Q_vol + Dirichlet BC contributions
rhs_base = -Q_int.ravel().copy()
rhs_base[:n_x] -= Ds[0, :].ravel() * T_BOT       # bottom wall
rhs_base[-n_x:] -= Dn[-1, :].ravel() * T_TOP      # top wall


def solve_temperature(u_field, v_field):
    """Solve the energy equation directly for given velocity field."""
    uc = u_field[1:-1, 1:-1]
    vc = v_field[1:-1, 1:-1]

    # Upwind convection half-coefficients  (zero in solid cells)
    Fw = RHO * CP * np.maximum(uc, 0.0) / dx     # west
    Fe = RHO * CP * np.maximum(-uc, 0.0) / dx    # east
    Fs = RHO * CP * np.maximum(vc, 0.0) / dy     # south
    Fn = RHO * CP * np.maximum(-vc, 0.0) / dy    # north
    Fp = RHO * CP * (np.abs(uc) / dx + np.abs(vc) / dy)

    # Build convection contribution to the matrix
    rows_c, cols_c, vals_c = [], [], []

    # Centre: subtract total convection from diagonal
    rows_c.append(kk.ravel())
    cols_c.append(kk.ravel())
    vals_c.append(-Fp.ravel())

    # East
    m = ((ii_int < n_x - 1) & (Fe > 0)).ravel()
    rows_c.append(kk.ravel()[m]);  cols_c.append((kk + 1).ravel()[m])
    vals_c.append(Fe.ravel()[m])

    # West
    m = ((ii_int > 0) & (Fw > 0)).ravel()
    rows_c.append(kk.ravel()[m]);  cols_c.append((kk - 1).ravel()[m])
    vals_c.append(Fw.ravel()[m])

    # North
    m = ((jj_int < n_y - 1) & (Fn > 0)).ravel()
    rows_c.append(kk.ravel()[m]);  cols_c.append((kk + n_x).ravel()[m])
    vals_c.append(Fn.ravel()[m])

    # South
    m = ((jj_int > 0) & (Fs > 0)).ravel()
    rows_c.append(kk.ravel()[m]);  cols_c.append((kk - n_x).ravel()[m])
    vals_c.append(Fs.ravel()[m])

    rc = np.concatenate(rows_c)
    cc = np.concatenate(cols_c)
    vc_ = np.concatenate(vals_c)
    A_conv = coo_matrix((vc_, (rc, cc)), shape=(n_T, n_T)).tocsc()

    A_total = A_diff + A_conv
    T_vec = spsolve(A_total, rhs_base)
    return T_vec.reshape(n_y, n_x)

# ================================================================
# 7. POISSON SOLVER (fluid-only stream function)
# ================================================================

print("Building Poisson matrix ...")

fluid_idx = -np.ones((NY, NX), dtype=int)
counter = 0
for j in range(1, NY - 1):
    for i in range(1, NX - 1):
        if fluid_int[j, i]:
            fluid_idx[j, i] = counter
            counter += 1
N_FLUID = counter

A_psi = lil_matrix((N_FLUID, N_FLUID))
for j in range(1, NY - 1):
    for i in range(1, NX - 1):
        if not fluid_int[j, i]:
            continue
        k = fluid_idx[j, i]
        A_psi[k, k] = -2.0 / dx2 - 2.0 / dy2
        for dj, di, h2 in [(0, 1, dx2), (0, -1, dx2), (1, 0, dy2), (-1, 0, dy2)]:
            nk = fluid_idx[j + dj, i + di]
            if nk >= 0:
                A_psi[k, nk] = 1.0 / h2

poisson_lu = splu(A_psi.tocsc())
del A_psi
fj, fi = np.where(fluid_int)

# ================================================================
# 8. THOM INTERFACE MASKS
# ================================================================

thom_above = np.zeros((NY, NX), dtype=bool)
thom_below = np.zeros((NY, NX), dtype=bool)
thom_right = np.zeros((NY, NX), dtype=bool)
thom_left = np.zeros((NY, NX), dtype=bool)

thom_above[:-1, :] = is_solid[:-1, :] & is_fluid[1:, :]
thom_below[1:, :] = is_solid[1:, :] & is_fluid[:-1, :]
thom_right[:, :-1] = is_solid[:, :-1] & is_fluid[:, 1:]
thom_left[:, 1:] = is_solid[:, 1:] & is_fluid[:, :-1]

is_solid_iface = thom_above | thom_below | thom_right | thom_left
thom_count = np.maximum(
    thom_above.astype(float) + thom_below.astype(float) +
    thom_right.astype(float) + thom_left.astype(float), 1.0)
is_solid_interior = is_solid & ~is_solid_iface

# Buoyancy mask: dT/dx only where both x-neighbours are fluid
buoy_ok = (fluid_int[1:-1, 1:-1] &
           is_fluid[1:-1, 2:] &
           is_fluid[1:-1, :-2])

# ================================================================
# 9. INITIALISE FIELDS
# ================================================================

T = T_BOT + DT_WALL * (Y / H)
psi = np.zeros((NY, NX))
omega = np.zeros((NY, NX))
u = np.zeros((NY, NX))
v = np.zeros((NY, NX))

# ================================================================
# 10. SOLVE
# ================================================================

MAX_ITER = 3000
TOL = 1e-4
RELAX_W = 0.15
N_SUB_W = 10

RA = GRAV * BETA * DT_WALL * H**3 / (NU * ALPHA_AIR)
fluid_mask_int = fluid_int[1:-1, 1:-1]

print("=" * 62)
print("  MacBook Cross-Section — 2D Natural Convection CFD")
print("=" * 62)
print(f"  Domain : {L*1e3:.0f} mm x {H*1e3:.0f} mm    Grid: {NX} x {NY}")
print(f"  T_bot = {T_BOT} C   T_top = {T_TOP} C")
print(f"  Ra (wall DT) = {RA:.1f}   Pr = {NU/ALPHA_AIR:.3f}")
print(f"  Fluid cells = {N_FLUID}   Solid cells = {int(is_solid.sum())}")
print(f"  Total heat  = {sum(c[5] for c in COMPONENTS):.1f} W/m")
print("=" * 62)

# --- Phase 1: Pure conduction (direct solve, no flow) ---
print("\n  Phase 1: conduction solve (direct) ...")
T[1:-1, 1:-1] = solve_temperature(u, v)
T[0, :] = T_BOT;  T[-1, :] = T_TOP
T[:, 0] = T[:, 1]; T[:, -1] = T[:, -2]
print(f"    T range = [{T.min():.2f}, {T.max():.2f}] C")

# --- Phase 2: Coupled iterations ---
print("\n  Phase 2: coupled solver ...")
converged = False

for iteration in range(1, MAX_ITER + 1):
    T_prev = T.copy()
    omega_prev = omega.copy()

    # A. Poisson
    psi[fj, fi] = poisson_lu.solve(-omega[fj, fi])

    # B. Velocity
    u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dy)
    v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dx)
    u[is_solid] = 0.0;  v[is_solid] = 0.0

    # C. Wall vorticity (enclosure)
    omega[0, :] = -2.0 * psi[1, :] / dy2
    omega[-1, :] = -2.0 * psi[-2, :] / dy2
    omega[:, 0] = -2.0 * psi[:, 1] / dx2
    omega[:, -1] = -2.0 * psi[:, -2] / dx2

    # D. Internal Thom
    omega_t = np.zeros((NY, NX))
    ps = np.zeros((NY, NX))
    ps[:-1, :] = psi[1:, :];   omega_t += np.where(thom_above, -2*ps/dy2, 0)
    ps[:] = 0; ps[1:, :] = psi[:-1, :]; omega_t += np.where(thom_below, -2*ps/dy2, 0)
    ps[:] = 0; ps[:, :-1] = psi[:, 1:]; omega_t += np.where(thom_right, -2*ps/dx2, 0)
    ps[:] = 0; ps[:, 1:] = psi[:, :-1]; omega_t += np.where(thom_left, -2*ps/dx2, 0)
    omega[is_solid_iface] = (omega_t / thom_count)[is_solid_iface]
    omega[is_solid_interior] = 0.0

    # E. Vorticity transport (iterative, fluid only, upwind)
    uc = u[1:-1, 1:-1];  vc = v[1:-1, 1:-1]
    up = np.maximum(uc, 0); um = np.minimum(uc, 0)
    vp = np.maximum(vc, 0); vm = np.minimum(vc, 0)

    aW_v = NU/dx2 + up/dx;  aE_v = NU/dx2 - um/dx
    aS_v = NU/dy2 + vp/dy;  aN_v = NU/dy2 - vm/dy
    den_v = aW_v + aE_v + aS_v + aN_v

    buoy = np.where(buoy_ok,
                    GRAV * BETA * (T[1:-1, 2:] - T[1:-1, :-2]) / (2*dx), 0.0)

    for _ in range(N_SUB_W):
        w_jac = (aW_v * omega[1:-1, :-2] + aE_v * omega[1:-1, 2:] +
                 aS_v * omega[:-2, 1:-1] + aN_v * omega[2:, 1:-1] +
                 buoy) / den_v
        omega[1:-1, 1:-1] = np.where(
            fluid_mask_int,
            (1 - RELAX_W) * omega[1:-1, 1:-1] + RELAX_W * w_jac,
            omega[1:-1, 1:-1])

    # F. Temperature (direct solve with current velocities)
    T[1:-1, 1:-1] = solve_temperature(u, v)
    T[0, :] = T_BOT;  T[-1, :] = T_TOP
    T[:, 0] = T[:, 1]; T[:, -1] = T[:, -2]

    # G. Convergence
    err_T = np.max(np.abs(T - T_prev))
    err_w = np.max(np.abs(omega - omega_prev))

    if iteration % 100 == 0 or iteration == 1:
        sp = np.sqrt(u**2 + v**2)
        print(f"    iter {iteration:5d}: dT={err_T:.3e} dw={err_w:.3e}"
              f"  T_peak={T.max():.1f}C  Vmax={sp.max()*1e3:.2f}mm/s")

    if iteration > 20 and max(err_T, err_w) < TOL:
        print(f"\n    Converged at iteration {iteration}")
        converged = True
        break

if not converged:
    print(f"\n    Stopped at {MAX_ITER} iterations (dT={err_T:.2e} dw={err_w:.2e})")

# ================================================================
# 11. POST-PROCESSING
# ================================================================

peak_T = T.max()
pj, pi = np.unravel_index(T.argmax(), T.shape)
peak_x, peak_y = x[pi] * 1e3, y[pj] * 1e3

speed = np.sqrt(u**2 + v**2)
v_max = np.where(is_fluid, speed, 0).max()

dTdy = (-3*T[0, :] + 4*T[1, :] - T[2, :]) / (2*dy)
Nu_avg = np.mean((H / DT_WALL) * dTdy)

DT_eff = max(peak_T - 0.5 * (T_BOT + T_TOP), 0.01)
Ra_eff = GRAV * BETA * DT_eff * H**3 / (NU * ALPHA_AIR)

print()
print("=" * 62)
print("  RESULTS")
print("=" * 62)
print(f"  Rayleigh number (wall DT)  : {RA:.1f}")
print(f"  Rayleigh number (effective): {Ra_eff:.1f}")
print(f"  Max air velocity           : {v_max * 1e3:.4f} mm/s")
print(f"  Avg Nusselt (bottom wall)  : {Nu_avg:.4f}")
print(f"  Peak temperature           : {peak_T:.2f} C  at ({peak_x:.1f}, {peak_y:.1f}) mm")
print("=" * 62)

# ================================================================
# 12. VISUALISATION
# ================================================================

x_mm, y_mm = x * 1e3, y * 1e3
X_mm, Y_mm = X * 1e3, Y * 1e3
fig, axes = plt.subplots(3, 1, figsize=(18, 10), constrained_layout=True)


def draw_outlines(ax, color="white", lw=1.0):
    for name, x0, x1, y0, y1, *_ in COMPONENTS:
        rect = patches.Rectangle((x0, y0), x1-x0, y1-y0,
                                 lw=lw, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text((x0+x1)/2, (y0+y1)/2, name, color=color,
                fontsize=5.5, ha="center", va="center", fontweight="bold")


# (a) Temperature
ax = axes[0]
lv = np.linspace(T.min(), T.max(), 30)
cf = ax.contourf(X_mm, Y_mm, T, levels=lv, cmap="inferno")
fig.colorbar(cf, ax=ax, label="Temperature (C)", pad=0.01)
draw_outlines(ax, "white", 0.9)
ax.set_title("Temperature Field"); ax.set_ylabel("y (mm)")
ax.set_xlim(0, L*1e3); ax.set_ylim(0, H*1e3)

# (b) Streamlines + speed
ax = axes[1]
cf = ax.contourf(X_mm, Y_mm, speed*1e3, 25, cmap="viridis")
fig.colorbar(cf, ax=ax, label="Speed (mm/s)", pad=0.01)
ax.streamplot(x_mm, y_mm, u, v, color="white", linewidth=0.5,
              density=(3, 1), arrowsize=0.8)
draw_outlines(ax, "cyan", 0.9)
ax.set_title("Velocity Streamlines"); ax.set_ylabel("y (mm)")
ax.set_xlim(0, L*1e3); ax.set_ylim(0, H*1e3)

# (c) Vorticity
ax = axes[2]
wa = max(abs(omega.min()), abs(omega.max()), 1e-12)
lv = np.linspace(-wa, wa, 30)
cf = ax.contourf(X_mm, Y_mm, omega, levels=lv, cmap="RdBu_r")
fig.colorbar(cf, ax=ax, label="Vorticity (1/s)", pad=0.01)
draw_outlines(ax, "black", 0.9)
ax.set_title("Vorticity Field"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_xlim(0, L*1e3); ax.set_ylim(0, H*1e3)

fig.suptitle(
    f"MacBook Natural Convection  |  Ra(wall)={RA:.0f}  Nu={Nu_avg:.2f}"
    f"  Vmax={v_max*1e3:.2f}mm/s  T_peak={peak_T:.1f}C",
    fontsize=11, fontweight="bold")
plt.savefig("macbook_natural_convection.png", dpi=150, bbox_inches="tight")
print(f"\n  Figure saved: macbook_natural_convection.png")
plt.close()
