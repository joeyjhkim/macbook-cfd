"""
MacBook Pro 16" — 2D Top-Down Forced Convection CFD (v3)
=========================================================
Corrected geometry: fans LEFT/RIGHT of SoC at same y-level,
vapor chamber arcs over SoC connecting both fans.
Non-overlapping components, fan forcing, k_field conduction.
"""

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import splu, spsolve
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# === 1. DOMAIN & GRID ===================================================

L, H = 0.355, 0.245
NX, NY = 260, 180
dx, dy = L / (NX - 1), H / (NY - 1)
dx2, dy2 = dx**2, dy**2
x = np.linspace(0, L, NX)
y = np.linspace(0, H, NY)
X, Y = np.meshgrid(x, y)

V_IN    = 1.2
T_IN    = 25.0
T_WALL  = 28.0
DEPTH_ACTIVE  = 0.130   # active zone: effective depth (3D z-spreading in SoC)
DEPTH_BATTERY = 0.055   # battery zone: accounts for chassis conduction in z
V_FAN   = 1.5

X_IN_L, X_IN_R = 0.080, 0.275
Y_OUT_B, Y_OUT_T = 0.200, 0.235

RHO, MU    = 1.109, 1.96e-5
K_AIR, CP  = 0.02756, 1007.0
NU = MU / RHO

j_inlet = int(round(0.120 / dy))
j_ob    = int(round(Y_OUT_B / dy))
j_ot    = int(round(Y_OUT_T / dy))

# === 2. COMPONENT DEFINITIONS ==========================================
# (name, x0mm, x1mm, y0mm, y1mm, Q_W, k)

COMPONENTS = [
    ("SoC",            162, 193, 158, 182, 15.0,  150),
    ("RAM_L",          140, 160, 162, 178,  2.0,  150),
    ("RAM_R",          195, 215, 162, 178,  2.0,  150),
    ("Fan_L",           88, 138, 158, 218,  0.5,   50),
    ("Fan_R",          217, 267, 158, 218,  0.5,   50),
    ("VC_L",           140, 162, 182, 218,  0.0, 2000),
    ("VC_R",           193, 215, 182, 218,  0.0, 2000),
    ("VC_top",         140, 215, 218, 228,  0.0, 2000),
    ("SSD_L",          148, 188, 230, 242,  1.5,  150),
    ("SSD_R",          192, 232, 230, 242,  1.5,  150),
    ("TB_L",             5,  25, 155, 175,  0.8,  150),
    ("TB_R",           330, 350, 155, 175,  0.8,  150),
    ("PowerCtrl",      275, 310, 158, 175,  1.0,  150),
    ("LogicBoard",      88, 267, 122, 157,  0.8,   80),
    ("Bat_A1",          42, 110,  78, 118,  1.2,    5),
    ("Bat_A2",          42, 110,  22,  76,  1.2,    5),
    ("Bat_B1",         118, 198,  22, 118,  2.0,    5),
    ("Bat_B2",         200, 250,  22, 118,  2.0,    5),
    ("Bat_C1",         252, 318,  78, 118,  1.2,    5),
    ("Bat_C2",         252, 318,  22,  76,  1.2,    5),
    ("Speaker_L",        5,  30,  10, 110,  0.2,   10),
    ("Speaker_R",      325, 350,  10, 110,  0.2,   10),
]

# === 3. OVERLAP CHECK ===================================================

print("=== Overlap Check ===")
overlaps = []
for i in range(len(COMPONENTS)):
    n1, ax0, ax1, ay0, ay1, *_ = COMPONENTS[i]
    for j in range(i + 1, len(COMPONENTS)):
        n2, bx0, bx1, by0, by1, *_ = COMPONENTS[j]
        if ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0:
            overlaps.append((n1, n2))
            print(f"  OVERLAP: {n1} <-> {n2}")
if not overlaps:
    print("  No overlaps detected.")
else:
    raise ValueError(f"{len(overlaps)} overlapping pairs found — fix layout!")

# === 4. SOLID MASK, k_field, Q_field ====================================

is_solid = np.zeros((NY, NX), dtype=bool)
is_fan   = np.zeros((NY, NX), dtype=bool)
comp_id  = -np.ones((NY, NX), dtype=int)
k_field  = np.full((NY, NX), K_AIR)
Q_field  = np.zeros((NY, NX))

for ci, (name, x0, x1, y0, y1, Q, k) in enumerate(COMPONENTS):
    mask = ((X >= x0 * 1e-3) & (X <= x1 * 1e-3) &
            (Y >= y0 * 1e-3) & (Y <= y1 * 1e-3))
    is_solid[mask] = True
    comp_id[mask]  = ci
    k_field[mask]  = k
    if Q > 0:
        area = (x1 - x0) * 1e-3 * (y1 - y0) * 1e-3
        # Battery zone uses physical depth; active zone uses effective depth
        depth = DEPTH_BATTERY if y1 <= 120 else DEPTH_ACTIVE
        Q_field[mask] = Q / (area * depth)
    if name.startswith("Fan"):
        is_fan[mask] = True

is_fluid = ~is_solid
print(f"  Solid: {int(is_solid.sum())}  Fluid: {int(is_fluid.sum())}  Fan: {int(is_fan.sum())}")

# === 5. LAYOUT VALIDATION FIGURE ========================================

print("Generating layout_validation.png ...")
cmap20 = plt.cm.tab20(np.linspace(0, 1, len(COMPONENTS)))

fig, ax = plt.subplots(figsize=(15, 10.5))
ax.add_patch(patches.Rectangle((0, 0), 355, 245, lw=2,
             edgecolor="black", facecolor="white"))

for ci, (name, x0, x1, y0, y1, Q, k) in enumerate(COMPONENTS):
    c = cmap20[ci]
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                 lw=1, edgecolor=c, facecolor=c, alpha=0.55))
    w = x1 - x0; h = y1 - y0
    fs = min(9, max(6, min(w, h) / 4))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, name, fontsize=fs,
            ha="center", va="center", fontweight="bold", color="black")

ax.plot([80, 275], [120, 120], lw=2.5, color="blue", label="Inlet")
ax.plot([0, 0], [200, 235], lw=2.5, color="red")
ax.plot([355, 355], [200, 235], lw=2.5, color="red", label="Outlets")
ax.axhline(120, color="gray", ls="--", lw=0.5)
ax.set_xlim(-3, 358);  ax.set_ylim(-3, 248)
ax.set_xlabel("x (mm)");  ax.set_ylabel("y (mm)")
ax.set_title("Component Layout — Overlap-Free Validation", fontsize=13)
ax.legend(loc="upper right", fontsize=8)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("layout_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved.")

# === 6. POTENTIAL FLOW ==================================================

print("Solving potential flow ...")
PSI_TOTAL = -V_IN * (X_IN_R - X_IN_L)
PSI_HALF  = PSI_TOTAL / 2.0

def inlet_psi(xi):
    if xi <= X_IN_L: return 0.0
    if xi >= X_IN_R: return PSI_TOTAL
    return -V_IN * (xi - X_IN_L)

psi_bc = np.zeros((NY, NX))
psi_bc[-1, :] = PSI_HALF
for j in range(NY):
    if y[j] > Y_OUT_T: psi_bc[j, 0] = PSI_HALF
for j in range(NY):
    yj = y[j]
    if   yj < 0.120:   psi_bc[j, -1] = 0.0
    elif yj < Y_OUT_B:  psi_bc[j, -1] = PSI_TOTAL
    elif yj <= Y_OUT_T:  psi_bc[j, -1] = PSI_TOTAL
    else:                psi_bc[j, -1] = PSI_HALF
for i in range(NX):
    psi_bc[j_inlet, i] = inlet_psi(x[i])

is_dir = np.zeros((NY, NX), dtype=bool)
is_dir[0, :] = True;  is_dir[-1, :] = True
is_dir[:, 0] = True;  is_dir[:, -1] = True
is_dir[j_inlet, :] = True

is_out_L = np.zeros(NY, dtype=bool)
is_out_R = np.zeros(NY, dtype=bool)
for j in range(NY):
    if j_ob <= j <= j_ot:
        is_out_L[j] = True;  is_dir[j, 0]  = False
        is_out_R[j] = True;  is_dir[j, -1] = False

# Solve Laplace through entire domain (including solid interiors)
psi_uk = np.zeros((NY, NX), dtype=bool)
psi_uk[1:-1, 1:-1] = True
psi_uk[j_inlet, :] = False
psi_uk &= ~is_dir

pidx = -np.ones((NY, NX), dtype=int)
c = 0
for j in range(NY):
    for i in range(NX):
        if psi_uk[j, i]: pidx[j, i] = c; c += 1
N_P = c

A_p = lil_matrix((N_P, N_P))
rhs_p = np.zeros(N_P)
for j in range(NY):
    for i in range(NX):
        if not psi_uk[j, i]: continue
        k = pidx[j, i]
        A_p[k, k] = -2 / dx2 - 2 / dy2
        for dj, di, h2 in [(0, 1, dx2), (0, -1, dx2), (1, 0, dy2), (-1, 0, dy2)]:
            nj, ni = j + dj, i + di
            if 0 <= nj < NY and 0 <= ni < NX:
                nk = pidx[nj, ni]
                if nk >= 0:          A_p[k, nk] = 1 / h2
                elif is_dir[nj, ni]: rhs_p[k] -= psi_bc[nj, ni] / h2

psi = psi_bc.copy()
uj, ui = np.where(psi_uk)
psi[uj, ui] = splu(A_p.tocsc()).solve(rhs_p)
del A_p
for j in range(NY):
    if is_out_L[j]: psi[j, 0] = psi[j, 1]
    if is_out_R[j]: psi[j, -1] = psi[j, -2]

# Velocity from psi, then enforce constraints
u = np.zeros((NY, NX));  v = np.zeros((NY, NX))
u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dy)
v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dx)

# Zero velocity in non-fan solids
u[is_solid & ~is_fan] = 0;  v[is_solid & ~is_fan] = 0

# Fan forcing: fans push air upward
u[is_fan] = 0;  v[is_fan] = V_FAN

# Inlet velocity
for i in range(NX):
    if X_IN_L <= x[i] <= X_IN_R: u[j_inlet, i] = 0; v[j_inlet, i] = V_IN

# Clip extreme velocities near psi-jump boundaries
speed = np.sqrt(u**2 + v**2)
V_CLIP = 4.0 * V_IN
clip = np.where(speed > V_CLIP, V_CLIP / np.maximum(speed, 1e-30), 1.0)
u *= clip;  v *= clip
speed = np.sqrt(u**2 + v**2)
print(f"  Vmax (clipped) = {speed.max():.2f} m/s")

# Vorticity (post-hoc)
omega = np.zeros((NY, NX))
omega[1:-1, 1:-1] = ((v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx) -
                      (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy))
omega[is_solid & ~is_fan] = 0

# === 7. TEMPERATURE SOLVE ===============================================

print("Building & solving temperature ...")

# Face conductivities using k_field (harmonic mean)
kx_f = np.zeros((NY, NX))
kx_f[:, :-1] = 2 * k_field[:, :-1] * k_field[:, 1:] / (k_field[:, :-1] + k_field[:, 1:])
ky_f = np.zeros((NY, NX))
ky_f[:-1, :] = 2 * k_field[:-1, :] * k_field[1:, :] / (k_field[:-1, :] + k_field[1:, :])

# Interior face conductances
De = kx_f[1:-1, 1:-1].copy() / dx2;  De[:, -1] = 0       # right Neumann
Dw = np.zeros((NY - 2, NX - 2));  Dw[:, 1:] = kx_f[1:-1, 1:-2] / dx2
Dn = ky_f[1:-1, 1:-1].copy() / dy2;  Dn[-1, :] = 0       # top Neumann
Ds = np.zeros((NY - 2, NX - 2))
Ds[1:, :] = ky_f[1:-2, 1:-1] / dy2
Ds[0, :]  = k_field[1, 1:-1] / dy2                         # bottom Dirichlet

n_x, n_y = NX - 2, NY - 2
n_T = n_x * n_y
jj, ii = np.mgrid[0:n_y, 0:n_x]
kk = jj * n_x + ii

# Diffusion matrix
rd, cd, vd = [], [], []
rd.append(kk.ravel());  cd.append(kk.ravel())
vd.append(-(De + Dw + Dn + Ds).ravel())

for msk, off, D in [((De > 0), 1, De), ((Dw > 0), -1, Dw),
                     ((jj < n_y - 1), n_x, Dn), ((jj > 0), -n_x, Ds)]:
    m = msk.ravel()
    rd.append(kk.ravel()[m]);  cd.append((kk + off).ravel()[m])
    vd.append(D.ravel()[m])

A_diff = coo_matrix((np.concatenate(vd),
                      (np.concatenate(rd), np.concatenate(cd))),
                     shape=(n_T, n_T)).tocsc()

# RHS: heat sources + bottom wall Dirichlet
rhs_T = -Q_field[1:-1, 1:-1].ravel().copy()
rhs_T[:n_x] -= Ds[0, :].ravel() * T_WALL

# Pin inlet vent cells to T_IN
inlet_mask = np.zeros((n_y, n_x), dtype=bool)
j_ii = j_inlet - 1
if 0 <= j_ii < n_y:
    for ic in range(n_x):
        if X_IN_L <= x[ic + 1] <= X_IN_R:
            inlet_mask[j_ii, ic] = True

A_diff = A_diff.tolil()
for idx in np.where(inlet_mask.ravel())[0]:
    A_diff[idx, :] = 0;  A_diff[idx, idx] = 1.0;  rhs_T[idx] = T_IN
A_diff = A_diff.tocsc()

# Convection (upwind) — corrected Fp diagonal
uc = u[1:-1, 1:-1].copy();  vc = v[1:-1, 1:-1].copy()
uc[inlet_mask] = 0;  vc[inlet_mask] = 0

Fw = RHO * CP * np.maximum(uc, 0) / dx
Fe = RHO * CP * np.maximum(-uc, 0) / dx
Fs = RHO * CP * np.maximum(vc, 0) / dy
Fn = RHO * CP * np.maximum(-vc, 0) / dy

offdiag_specs = [((ii < n_x - 1) & (Fe > 0), 1, Fe),
                 ((ii > 0) & (Fw > 0), -1, Fw),
                 ((jj < n_y - 1) & (Fn > 0), n_x, Fn),
                 ((jj > 0) & (Fs > 0), -n_x, Fs)]

# Fp = sum of ONLY the off-diag entries that actually exist
Fp = np.zeros((n_y, n_x))
for msk, _, Fc in offdiag_specs:
    Fp += np.where(msk, Fc, 0)

rc, cc, vc_ = [], [], []
rc.append(kk.ravel());  cc.append(kk.ravel());  vc_.append(-Fp.ravel())
for msk, off, Fc in offdiag_specs:
    m = msk.ravel()
    rc.append(kk.ravel()[m]);  cc.append((kk + off).ravel()[m])
    vc_.append(Fc.ravel()[m])

A_conv = coo_matrix((np.concatenate(vc_),
                      (np.concatenate(rc), np.concatenate(cc))),
                     shape=(n_T, n_T)).tocsc()

print("  Solving linear system ...")
T_int = spsolve(A_diff + A_conv, rhs_T).reshape(n_y, n_x)

# Assemble full T field
T = np.full((NY, NX), T_IN)
T[1:-1, 1:-1] = T_int
T[0, :] = T_WALL
T[:, 0] = T[:, 1];  T[:, -1] = T[:, -2]
for i in range(NX):
    if X_IN_L <= x[i] <= X_IN_R: T[j_inlet, i] = T_IN
for j in range(NY):
    if is_out_L[j]: T[j, 0]  = T[j, 1]
    if is_out_R[j]: T[j, -1] = T[j, -2]
T = np.clip(T, T_IN - 1, None)

print(f"  T range = [{T.min():.1f}, {T.max():.1f}] C")

# === 8. POST-PROCESSING =================================================

RE = RHO * V_IN * L / MU
v_max = speed.max()
peak_T = T.max()
pj, pi = np.unravel_index(T.argmax(), T.shape)

soc_m = (comp_id == 0)
T_soc = T[soc_m].mean() if soc_m.any() else 0

bat_m = np.zeros((NY, NX), dtype=bool)
for ci, (nm, *_) in enumerate(COMPONENTS):
    if nm.startswith("Bat"): bat_m |= (comp_id == ci)
T_bat = T[bat_m].max() if bat_m.any() else 0

Nu_soc = 0
if soc_m.any() and (T_soc - T_IN) > 0.1:
    c = COMPONENTS[0]
    A_s = (c[2] - c[1]) * 1e-3 * (c[4] - c[3]) * 1e-3
    Nu_soc = c[5] / (A_s * (T_soc - T_IN)) * (c[2] - c[1]) * 1e-3 / K_AIR

print()
print("=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"  Reynolds number     : {RE:.0f}")
print(f"  Max air velocity    : {v_max:.2f} m/s")
print(f"  Peak temperature    : {peak_T:.1f} C  at ({x[pi]*1e3:.1f}, {y[pj]*1e3:.1f}) mm")
print(f"  SoC avg temperature : {T_soc:.1f} C")
print(f"  Battery max temp    : {T_bat:.1f} C")
print(f"  Nusselt on SoC      : {Nu_soc:.1f}")
print("=" * 60)

# === 9-11. FIGURES ======================================================

x_mm, y_mm = x * 1e3, y * 1e3
X_mm, Y_mm = X * 1e3, Y * 1e3


def draw_outlines(ax, color="white", lw=0.7):
    for nm, x0, x1, y0, y1, *_ in COMPONENTS:
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                     lw=lw, edgecolor=color, facecolor="none"))
        lab = nm.replace("_", " ")
        # Size label based on component area so it fits
        w = x1 - x0;  h = y1 - y0
        fs = min(8, max(5.5, min(w, h) / 4.5))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, lab, color=color,
                fontsize=fs, ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4, lw=0))


def mark_io(ax):
    ax.plot([80, 275], [120, 120], lw=2.5, color="deepskyblue", zorder=5)
    ax.annotate("INLET v=1.2m/s", xy=(177, 118), fontsize=7,
                color="deepskyblue", ha="center", va="top", fontweight="bold")
    ax.plot([0, 0], [200, 235], lw=2.5, color="tomato", zorder=5)
    ax.plot([355, 355], [200, 235], lw=2.5, color="tomato", zorder=5)
    ax.annotate("OUT", xy=(4, 217), fontsize=6, color="tomato", fontweight="bold")
    ax.annotate("OUT", xy=(351, 217), fontsize=6, color="tomato",
                ha="right", fontweight="bold")


# --- Temperature ---
fig, ax = plt.subplots(figsize=(16, 11))
lv = np.linspace(T.min(), T.max(), 40)
cf = ax.contourf(X_mm, Y_mm, T, levels=lv, cmap="inferno")
fig.colorbar(cf, ax=ax, label="Temperature (°C)", pad=0.01, shrink=0.82)
draw_outlines(ax, "white");  mark_io(ax)
ax.set_title(f"Temperature  |  Re={RE:.0f}  Vmax={v_max:.1f}m/s  "
             f"T_peak={peak_T:.1f}°C  T_SoC={T_soc:.1f}°C  T_bat={T_bat:.1f}°C",
             fontsize=10, fontweight="bold")
ax.set_xlabel("x (mm)");  ax.set_ylabel("y (mm)")
ax.set_xlim(0, 355);  ax.set_ylim(0, 245)
plt.tight_layout()
plt.savefig("macbook_temperature.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Velocity ---
fig, ax = plt.subplots(figsize=(16, 11))
cf = ax.contourf(X_mm, Y_mm, speed, 30, cmap="viridis")
fig.colorbar(cf, ax=ax, label="Speed (m/s)", pad=0.01, shrink=0.82)
ax.streamplot(x_mm, y_mm, u, v, color="white", linewidth=0.8,
              density=2, arrowsize=0.8)
for nm, x0, x1, y0, y1, *_ in COMPONENTS:
    if nm.startswith("Fan"):
        cx = (x0 + x1) / 2;  cy = (y0 + y1) / 2
        ax.annotate("", xy=(cx, y1 - 3), xytext=(cx, y0 + 3),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=2.5))
        ax.text(cx, cy, "FAN", color="yellow", fontsize=7,
                ha="center", va="center", fontweight="bold")
draw_outlines(ax, "cyan");  mark_io(ax)
ax.set_title("Velocity Streamlines", fontsize=11, fontweight="bold")
ax.set_xlabel("x (mm)");  ax.set_ylabel("y (mm)")
ax.set_xlim(0, 355);  ax.set_ylim(0, 245)
plt.tight_layout()
plt.savefig("macbook_velocity.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Vorticity ---
fig, ax = plt.subplots(figsize=(16, 11))
wc = 25.0
lv = np.linspace(-wc, wc, 30)
cf = ax.contourf(X_mm, Y_mm, np.clip(omega, -wc, wc), levels=lv, cmap="RdBu_r")
fig.colorbar(cf, ax=ax, label="Vorticity (1/s)", pad=0.01, shrink=0.82)
draw_outlines(ax, "black");  mark_io(ax)
ax.set_title("Vorticity Field", fontsize=11, fontweight="bold")
ax.set_xlabel("x (mm)");  ax.set_ylabel("y (mm)")
ax.set_xlim(0, 355);  ax.set_ylim(0, 245)
plt.tight_layout()
plt.savefig("macbook_vorticity.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n  Figures saved:")
print("    layout_validation.png")
print("    macbook_temperature.png")
print("    macbook_velocity.png")
print("    macbook_vorticity.png")
