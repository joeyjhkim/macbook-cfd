"""
2D Natural Convection in a Sealed Rectangular Enclosure
=======================================================
Stream function-vorticity formulation with Boussinesq approximation.

Physics:
  Enclosure: 20 mm wide x 10 mm tall (sealed wearable device)
  Left wall:  60 C (chip)        Right wall: 35 C (ambient sink)
  Top/bottom: adiabatic (dT/dn = 0)
  Gravity:    -y direction
  Fluid:      air at mean temp 47.5 C

Governing equations (steady-state):
  1. Poisson:              nabla^2 psi = -omega
  2. Vorticity transport:  u dw/dx + v dw/dy = nu nabla^2 w + g beta dT/dx
  3. Energy:               u dT/dx + v dT/dy = alpha nabla^2 T

  Velocity recovery: u = dpsi/dy,  v = -dpsi/dx
  Boussinesq: density variation appears only in the buoyancy term (g beta dT/dx).

Numerical method:
  - Direct sparse solve (LU-factored) for the Poisson equation each iteration
  - Vectorised Jacobi sweeps with under-relaxation for vorticity and energy
  - Central differences for all spatial derivatives (cell Peclet < 2)
  - Thom's formula for wall vorticity (second-order accurate)
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt

# ================================================================
# 1. PHYSICAL PARAMETERS
# ================================================================

L = 0.020          # enclosure width  [m]  (20 mm)
H = 0.010          # enclosure height [m]  (10 mm)

T_HOT = 60.0       # left wall temperature  [C] (chip)
T_COLD = 35.0      # right wall temperature [C] (ambient sink)
DT = T_HOT - T_COLD
T_REF = 0.5 * (T_HOT + T_COLD)  # 47.5 C reference / mean temperature

# Air properties evaluated at T_REF ~ 47.5 C (320.65 K)
NU = 1.75e-5       # kinematic viscosity   [m^2/s]
ALPHA = 2.50e-5    # thermal diffusivity   [m^2/s]
BETA = 1.0 / (T_REF + 273.15)   # vol. expansion coeff [1/K] (ideal gas)
G = 9.81           # gravitational acceleration [m/s^2]

# Derived dimensionless numbers
PR = NU / ALPHA                        # Prandtl number
RA = G * BETA * DT * H**3 / (NU * ALPHA)  # Rayleigh number (height-based)

# ================================================================
# 2. COMPUTATIONAL GRID
# ================================================================

NX, NY = 60, 30            # grid points in x, y
dx = L / (NX - 1)
dy = H / (NY - 1)
dx2, dy2 = dx * dx, dy * dy

x = np.linspace(0, L, NX)
y = np.linspace(0, H, NY)
X, Y = np.meshgrid(x, y)   # shape (NY, NX); X[j, i], Y[j, i]

# Pre-computed diffusion coefficients on the grid
ax = ALPHA / dx2             # thermal diffusion coefficient, x
ay = ALPHA / dy2             # thermal diffusion coefficient, y
vx = NU / dx2                # viscous diffusion coefficient, x
vy = NU / dy2                # viscous diffusion coefficient, y
denom_T = 2.0 * (ax + ay)   # diagonal coefficient for energy Jacobi
denom_w = 2.0 * (vx + vy)   # diagonal coefficient for vorticity Jacobi

# ================================================================
# 3. BUILD POISSON SOLVER FOR STREAM FUNCTION
# ================================================================
# Assemble the discrete 2D Laplacian for interior points and
# LU-factorise once.  psi = 0 on all four walls (no-slip +
# single-valued stream function on a closed boundary).

n_ix = NX - 2   # interior point count, x-direction
n_iy = NY - 2   # interior point count, y-direction
N_INT = n_ix * n_iy

A_psi = lil_matrix((N_INT, N_INT))

for jj in range(n_iy):
    for ii in range(n_ix):
        k = jj * n_ix + ii                     # linear index
        A_psi[k, k] = -2.0 / dx2 - 2.0 / dy2  # centre coefficient

        if ii > 0:          A_psi[k, k - 1]    = 1.0 / dx2   # left
        if ii < n_ix - 1:   A_psi[k, k + 1]    = 1.0 / dx2   # right
        if jj > 0:          A_psi[k, k - n_ix]  = 1.0 / dy2  # below
        if jj < n_iy - 1:   A_psi[k, k + n_ix]  = 1.0 / dy2  # above

# Pre-factor for fast repeated solves (factorisation is O(N^1.5))
poisson_lu = splu(A_psi.tocsc())
del A_psi

# ================================================================
# 4. INITIALISE FIELDS
# ================================================================

# Temperature: linear profile from hot to cold (good starting guess)
T = np.outer(np.ones(NY), np.linspace(T_HOT, T_COLD, NX))

# Stream function and vorticity: quiescent initial state
psi = np.zeros((NY, NX))
omega = np.zeros((NY, NX))

# Velocity components
u = np.zeros((NY, NX))   # horizontal (x-dir)
v = np.zeros((NY, NX))   # vertical   (y-dir)

# ================================================================
# 5. ITERATIVE SOLVER
# ================================================================

MAX_ITER = 25_000
TOL = 1e-6
RELAX_W = 0.7    # under-relaxation for vorticity (stabilises psi-omega coupling)
RELAX_T = 0.9    # under-relaxation for temperature

print("=" * 58)
print("  2D Natural Convection  -  Sealed Enclosure")
print("=" * 58)
print(f"  Domain:   {L*1e3:.0f} mm x {H*1e3:.0f} mm   Grid: {NX} x {NY}")
print(f"  T_hot:    {T_HOT:.1f} C     T_cold: {T_COLD:.1f} C")
print(f"  Ra = {RA:.1f}    Pr = {PR:.3f}    AR = {L/H:.1f}")
print("=" * 58)
print("  Iterating ...\n")

converged = False

for iteration in range(1, MAX_ITER + 1):
    # Snapshot for convergence check
    omega_prev = omega.copy()
    T_prev = T.copy()

    # ------------------------------------------------------------------
    # Step A: Solve Poisson equation  nabla^2 psi = -omega  (direct)
    # ------------------------------------------------------------------
    # The LU factorisation of the Laplacian was computed once in section 3.
    # Only the RHS (which depends on the current vorticity) changes.
    rhs = -omega[1:-1, 1:-1].ravel()
    psi[1:-1, 1:-1] = poisson_lu.solve(rhs).reshape(n_iy, n_ix)
    # Boundary: psi = 0 on all walls (already initialised, never modified)

    # ------------------------------------------------------------------
    # Step B: Recover velocity from stream function
    # ------------------------------------------------------------------
    # u = dpsi/dy  (central difference in y)
    u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dy)
    # v = -dpsi/dx (central difference in x)
    v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dx)
    # Walls: u = v = 0 (no-slip, already zero from initialisation)

    # ------------------------------------------------------------------
    # Step C: Wall vorticity via Thom's formula
    # ------------------------------------------------------------------
    # Derived from a Taylor expansion of psi near the wall combined with
    # no-slip (u = v = 0) and psi = 0 on the wall:
    #   omega_wall = -2 * psi_adjacent / h^2
    omega[0, :]  = -2.0 * psi[1, :]  / dy2     # bottom wall
    omega[-1, :] = -2.0 * psi[-2, :] / dy2     # top wall
    omega[:, 0]  = -2.0 * psi[:, 1]  / dx2     # left wall  (hot)
    omega[:, -1] = -2.0 * psi[:, -2] / dx2     # right wall (cold)

    # ------------------------------------------------------------------
    # Step D: Vorticity transport  (interior, one Jacobi sweep)
    # ------------------------------------------------------------------
    # Steady-state:  nu nabla^2 w - u dw/dx - v dw/dy + g beta dT/dx = 0
    #
    # Central FD for all derivatives.  Rearranged as a Jacobi update:
    #   w_P = [ (vx+cx)*w_W + (vx-cx)*w_E
    #         + (vy+cy)*w_S + (vy-cy)*w_N
    #         + buoyancy ] / denom_w
    #
    # where cx = u/(2dx), cy = v/(2dy) are the convection half-coefficients.
    cx = u[1:-1, 1:-1] / (2.0 * dx)
    cy = v[1:-1, 1:-1] / (2.0 * dy)

    # Buoyancy source term:  g * beta * dT/dx
    # Hot fluid on the left (dT/dx < 0) drives clockwise (negative) vorticity.
    buoyancy = G * BETA * (T[1:-1, 2:] - T[1:-1, :-2]) / (2.0 * dx)

    w_jacobi = (
        (vx + cx) * omega[1:-1, :-2]    # west  (i-1)
      + (vx - cx) * omega[1:-1, 2:]     # east  (i+1)
      + (vy + cy) * omega[:-2, 1:-1]    # south (j-1)
      + (vy - cy) * omega[2:, 1:-1]     # north (j+1)
      + buoyancy
    ) / denom_w

    # Under-relaxation:  w_new = (1 - alpha)*w_old + alpha*w_jacobi
    omega[1:-1, 1:-1] = (
        (1.0 - RELAX_W) * omega_prev[1:-1, 1:-1]
        + RELAX_W * w_jacobi
    )

    # ------------------------------------------------------------------
    # Step E: Energy equation  (interior, one Jacobi sweep)
    # ------------------------------------------------------------------
    # Steady-state:  alpha nabla^2 T - u dT/dx - v dT/dy = 0
    #
    # Same Jacobi structure as vorticity but without a source term.
    # cx, cy are identical (same velocity field) so we reuse them.
    T_jacobi = (
        (ax + cx) * T[1:-1, :-2]        # west  (i-1)
      + (ax - cx) * T[1:-1, 2:]         # east  (i+1)
      + (ay + cy) * T[:-2, 1:-1]        # south (j-1)
      + (ay - cy) * T[2:, 1:-1]         # north (j+1)
    ) / denom_T

    T[1:-1, 1:-1] = (
        (1.0 - RELAX_T) * T_prev[1:-1, 1:-1]
        + RELAX_T * T_jacobi
    )

    # ------------------------------------------------------------------
    # Step F: Apply temperature boundary conditions
    # ------------------------------------------------------------------
    T[:, 0]  = T_HOT        # left wall  (Dirichlet - chip surface)
    T[:, -1] = T_COLD       # right wall (Dirichlet - ambient sink)
    T[0, :]  = T[1, :]      # bottom wall (adiabatic: dT/dy = 0)
    T[-1, :] = T[-2, :]     # top wall    (adiabatic: dT/dy = 0)

    # ------------------------------------------------------------------
    # Step G: Convergence check
    # ------------------------------------------------------------------
    err_T = np.max(np.abs(T - T_prev))
    err_w = np.max(np.abs(omega - omega_prev))

    if iteration % 2000 == 0 or iteration == 1:
        print(f"    iter {iteration:6d}:  dT = {err_T:.3e}   dw = {err_w:.3e}")

    if iteration > 100 and err_T < TOL and err_w < TOL:
        print(f"\n    Converged at iteration {iteration}"
              f"  (dT = {err_T:.2e}, dw = {err_w:.2e})")
        converged = True
        break

if not converged:
    print(f"\n    Not converged after {MAX_ITER} iterations"
          f"  (dT = {err_T:.2e}, dw = {err_w:.2e})")

# ================================================================
# 6. POST-PROCESSING
# ================================================================

# Maximum velocity magnitude
speed = np.sqrt(u**2 + v**2)
v_max = np.max(speed)

# Average Nusselt number on the hot wall
# Use a second-order one-sided finite difference for dT/dx at x = 0:
#   dT/dx |_{x=0}  ~  (-3 T_0 + 4 T_1 - T_2) / (2 dx)
# For pure conduction: dT/dx = -DT/L, so Nu = 1.
# Convection enhances the gradient, giving Nu > 1.
dTdx_hot = (-3.0 * T[:, 0] + 4.0 * T[:, 1] - T[:, 2]) / (2.0 * dx)
Nu_avg = -(L / DT) * np.trapezoid(dTdx_hot, y) / H

print()
print("=" * 58)
print("  RESULTS")
print("=" * 58)
print(f"  Rayleigh number:       Ra   = {RA:.1f}")
print(f"  Max velocity:          Vmax = {v_max * 1e3:.4f} mm/s")
print(f"  Avg Nusselt number:    Nu   = {Nu_avg:.4f}")
print("=" * 58)

# ================================================================
# 7. VISUALISATION  (3 subplots)
# ================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x_mm = x * 1e3
y_mm = y * 1e3
X_mm = X * 1e3
Y_mm = Y * 1e3

# ------ (a) Temperature contour map ------
ax = axes[0]
levels_T = np.linspace(T_COLD, T_HOT, 26)
cf = ax.contourf(X_mm, Y_mm, T, levels=levels_T, cmap="inferno")
ax.contour(X_mm, Y_mm, T, levels=levels_T[::2], colors="w",
           linewidths=0.4, alpha=0.5)
fig.colorbar(cf, ax=ax, label="Temperature (C)", shrink=0.82)
ax.set_title("Temperature Field")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_aspect("equal")

# ------ (b) Velocity streamlines on speed magnitude ------
ax = axes[1]
speed_mm = speed * 1e3  # convert m/s -> mm/s
cf = ax.contourf(X_mm, Y_mm, speed_mm, 20, cmap="viridis")
ax.streamplot(x_mm, y_mm, u, v, color="white", linewidth=0.7,
              density=1.5, arrowsize=1.0)
fig.colorbar(cf, ax=ax, label="Speed (mm/s)", shrink=0.82)
ax.set_title("Velocity Streamlines")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_aspect("equal")

# ------ (c) Vorticity contour map ------
ax = axes[2]
w_abs_max = max(abs(omega.min()), abs(omega.max()))
if w_abs_max < 1e-12:
    w_abs_max = 1.0  # guard against zero field
levels_w = np.linspace(-w_abs_max, w_abs_max, 26)
cf = ax.contourf(X_mm, Y_mm, omega, levels=levels_w, cmap="RdBu_r")
ax.contour(X_mm, Y_mm, omega, levels=levels_w[::2], colors="k",
           linewidths=0.3, alpha=0.4)
fig.colorbar(cf, ax=ax, label="Vorticity (1/s)", shrink=0.82)
ax.set_title("Vorticity Field")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_aspect("equal")

fig.suptitle(
    f"Natural Convection in Sealed Enclosure   |   "
    f"Ra = {RA:.0f},  Nu = {Nu_avg:.2f},  "
    f"Vmax = {v_max*1e3:.2f} mm/s",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.savefig("natural_convection_enclosure.png", dpi=200, bbox_inches="tight")
print(f"\n  Figure saved: natural_convection_enclosure.png")
plt.close()
