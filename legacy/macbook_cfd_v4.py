"""
MacBook Pro 16" — 2D Top-Down Forced Convection CFD (v4)
=========================================================
Tuned to match real MacBook Pro thermal imaging data:
  - Rear hinge exhaust (top wall outlets, not sides)
  - Aluminum chassis ground plane for lateral heat spreading
  - SoC offset slightly left (matches real placement)
  - Calibrated Q/depth for each zone
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

V_IN  = 1.2;  T_IN = 25.0;  T_WALL = 28.0;  V_FAN = 1.5

# Inlet: bottom of active zone (air enters through bottom case vents)
X_IN_L, X_IN_R = 0.080, 0.275
# Outlets: TOP wall (rear hinge exhaust) — two bands
X_OL_L, X_OL_R = 0.080, 0.168   # left exhaust
X_OR_L, X_OR_R = 0.187, 0.275   # right exhaust

# Effective depth per zone (calibrated to match real thermal data)
DEPTH_ACTIVE  = 0.110   # SoC ~90°C target
DEPTH_BATTERY = 0.045   # Battery ~40°C target

RHO, MU   = 1.109, 1.96e-5
K_AIR, CP = 0.02756, 1007.0
NU = MU / RHO

# Slight chassis enhancement in battery zone only (batteries sit on Al plate)
# Active zone air stays at k_air (DEPTH calibration handles 3D effects)
K_CHASSIS_BATTERY = 0.05   # modest boost from Al plate contact (vs k_air=0.028)

j_inlet = int(round(0.120 / dy))

def ix(mm): return int(round(mm * 1e-3 / dx))
i_ol_l, i_ol_r = ix(80), ix(168)
i_or_l, i_or_r = ix(187), ix(275)

# === 2. COMPONENT DEFINITIONS ==========================================

COMPONENTS = [
    # name,          x0,  x1,  y0,  y1,    Q,     k
    # --- SoC offset ~7mm left of center (real placement) ---
    ("SoC",          155, 186, 158, 182,  15.0,   150),
    ("RAM_L",        133, 153, 162, 178,   2.0,   150),
    ("RAM_R",        188, 208, 162, 178,   2.0,   150),
    ("Fan_L",         82, 131, 158, 218,   0.5,    50),
    ("Fan_R",        210, 260, 158, 218,   0.5,    50),
    ("VC_L",         133, 155, 182, 218,   0.0,  2000),
    ("VC_R",         186, 208, 182, 218,   0.0,  2000),
    ("VC_top",       133, 208, 218, 224,   0.0,  2000),
    ("SSD_L",         32,  62, 140, 155,   0.4,   150),
    ("SSD_R",        293, 323, 140, 155,   0.4,   150),
    ("TB_L",           5,  25, 155, 175,   0.8,   150),
    ("TB_R",         330, 350, 155, 175,   0.8,   150),
    ("PowerCtrl",    268, 290, 158, 175,   1.0,   150),
    ("LogicBoard",    82, 260, 122, 157,   0.8,    80),
    ("Bat_A1",        42, 110,  78, 118,   1.2,     5),
    ("Bat_A2",        42, 110,  22,  76,   1.2,     5),
    ("Bat_B1",       118, 198,  22, 118,   2.0,     5),
    ("Bat_B2",       200, 250,  22, 118,   2.0,     5),
    ("Bat_C1",       252, 318,  78, 118,   1.2,     5),
    ("Bat_C2",       252, 318,  22,  76,   1.2,     5),
    ("Speaker_L",      5,  30,  10, 110,   0.2,    10),
    ("Speaker_R",    325, 350,  10, 110,   0.2,    10),
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
if overlaps:
    raise ValueError(f"{len(overlaps)} overlapping pairs!")
print("  No overlaps detected.")

# === 4. SOLID MASK, k_field, Q_field ====================================

is_solid = np.zeros((NY, NX), dtype=bool)
is_fan   = np.zeros((NY, NX), dtype=bool)
comp_id  = -np.ones((NY, NX), dtype=int)

# k_field: air everywhere, modest boost in battery zone for chassis contact
k_field = np.full((NY, NX), K_AIR)
k_field[Y < 0.120] = K_CHASSIS_BATTERY

Q_field = np.zeros((NY, NX))

for ci, (name, x0, x1, y0, y1, Q, k) in enumerate(COMPONENTS):
    mask = ((X >= x0 * 1e-3) & (X <= x1 * 1e-3) &
            (Y >= y0 * 1e-3) & (Y <= y1 * 1e-3))
    is_solid[mask] = True
    comp_id[mask]  = ci
    k_field[mask]  = k
    if Q > 0:
        area = (x1 - x0) * 1e-3 * (y1 - y0) * 1e-3
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
for ci, (name, x0, x1, y0, y1, *_) in enumerate(COMPONENTS):
    c = cmap20[ci]
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                 lw=1, edgecolor=c, facecolor=c, alpha=0.55))
    w, h = x1 - x0, y1 - y0
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, name,
            fontsize=min(9, max(6, min(w, h) / 4)),
            ha="center", va="center", fontweight="bold", color="black")
# Inlet / outlets
ax.plot([80, 275], [120, 120], lw=2.5, color="blue", label="Inlet (bottom vents)")
ax.plot([80, 168], [245, 245], lw=3, color="red")
ax.plot([187, 275], [245, 245], lw=3, color="red", label="Exhaust (rear hinge)")
ax.axhline(120, color="gray", ls="--", lw=0.5)
ax.set_xlim(-3, 358); ax.set_ylim(-3, 248)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_title("MacBook Pro 16\" Component Layout — Rear Exhaust", fontsize=13)
ax.legend(loc="upper right", fontsize=8)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("layout_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved.")

# === 6. POTENTIAL FLOW (rear exhaust) ===================================

print("Solving potential flow ...")
PSI_TOTAL = -V_IN * (X_IN_R - X_IN_L)
PSI_MID   = PSI_TOTAL / 2.0

def inlet_psi(xi):
    if xi <= X_IN_L: return 0.0
    if xi >= X_IN_R: return PSI_TOTAL
    return -V_IN * (xi - X_IN_L)

psi_bc = np.zeros((NY, NX))

# Bottom wall: psi = 0
# Left wall: psi = 0 everywhere (no side outlets)
# Right wall: 0 below inlet, PSI_TOTAL above inlet
psi_bc[j_inlet:, -1] = PSI_TOTAL

# Inlet row
for i in range(NX): psi_bc[j_inlet, i] = inlet_psi(x[i])

# Top wall: psi varies for exhaust bands
psi_bc[-1, :i_ol_l]          = 0.0
psi_bc[-1, i_ol_r + 1:i_or_l] = PSI_MID
psi_bc[-1, i_or_r + 1:]       = PSI_TOTAL

# Dirichlet mask
is_dir = np.zeros((NY, NX), dtype=bool)
is_dir[0, :] = True;  is_dir[-1, :] = True
is_dir[:, 0] = True;  is_dir[:, -1] = True
is_dir[j_inlet, :] = True

# Outlets at top wall → Neumann
is_outlet_top = np.zeros(NX, dtype=bool)
for i in range(NX):
    if (i_ol_l <= i <= i_ol_r) or (i_or_l <= i <= i_or_r):
        is_outlet_top[i] = True
        is_dir[-1, i] = False

# Build & solve Laplace (through entire domain including solids)
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
# Neumann at outlets
for i in range(NX):
    if is_outlet_top[i]: psi[-1, i] = psi[-2, i]

# Velocity + constraints
u = np.zeros((NY, NX));  v = np.zeros((NY, NX))
u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dy)
v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dx)
u[is_solid & ~is_fan] = 0;  v[is_solid & ~is_fan] = 0
u[is_fan] = 0;  v[is_fan] = V_FAN
for i in range(NX):
    if X_IN_L <= x[i] <= X_IN_R:
        u[j_inlet, i] = 0;  v[j_inlet, i] = V_IN

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

# Face conductivities (harmonic mean of k_field)
kx_f = np.zeros((NY, NX))
kx_f[:, :-1] = 2 * k_field[:, :-1] * k_field[:, 1:] / (k_field[:, :-1] + k_field[:, 1:])
ky_f = np.zeros((NY, NX))
ky_f[:-1, :] = 2 * k_field[:-1, :] * k_field[1:, :] / (k_field[:-1, :] + k_field[1:, :])

De = kx_f[1:-1, 1:-1].copy() / dx2;  De[:, -1] = 0
Dw = np.zeros((NY - 2, NX - 2));     Dw[:, 1:] = kx_f[1:-1, 1:-2] / dx2
Dn = ky_f[1:-1, 1:-1].copy() / dy2;  Dn[-1, :] = 0   # top Neumann
Ds = np.zeros((NY - 2, NX - 2))
Ds[1:, :] = ky_f[1:-2, 1:-1] / dy2
Ds[0, :]  = k_field[1, 1:-1] / dy2   # bottom Dirichlet

n_x, n_y = NX - 2, NY - 2;  n_T = n_x * n_y
jj, ii = np.mgrid[0:n_y, 0:n_x];  kk = jj * n_x + ii

# Diffusion matrix
rd, cd, vd = [], [], []
rd.append(kk.ravel()); cd.append(kk.ravel())
vd.append(-(De + Dw + Dn + Ds).ravel())
for msk, off, D in [((De > 0), 1, De), ((Dw > 0), -1, Dw),
                     ((jj < n_y - 1), n_x, Dn), ((jj > 0), -n_x, Ds)]:
    m = msk.ravel()
    rd.append(kk.ravel()[m]); cd.append((kk + off).ravel()[m]); vd.append(D.ravel()[m])
A_diff = coo_matrix((np.concatenate(vd), (np.concatenate(rd), np.concatenate(cd))),
                     shape=(n_T, n_T)).tocsc()

rhs_T = -Q_field[1:-1, 1:-1].ravel().copy()
rhs_T[:n_x] -= Ds[0, :].ravel() * T_WALL

# Pin inlet cells
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

# Upwind convection with corrected Fp
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
Fp = np.zeros((n_y, n_x))
for msk, _, Fc in offdiag_specs:
    Fp += np.where(msk, Fc, 0)

rc, cc, vc_ = [], [], []
rc.append(kk.ravel()); cc.append(kk.ravel()); vc_.append(-Fp.ravel())
for msk, off, Fc in offdiag_specs:
    m = msk.ravel()
    rc.append(kk.ravel()[m]); cc.append((kk + off).ravel()[m]); vc_.append(Fc.ravel()[m])
A_conv = coo_matrix((np.concatenate(vc_), (np.concatenate(rc), np.concatenate(cc))),
                     shape=(n_T, n_T)).tocsc()

T_int = spsolve(A_diff + A_conv, rhs_T).reshape(n_y, n_x)

T = np.full((NY, NX), T_IN)
T[1:-1, 1:-1] = T_int
T[0, :] = T_WALL;  T[:, 0] = T[:, 1];  T[:, -1] = T[:, -2]
for i in range(NX):
    if X_IN_L <= x[i] <= X_IN_R: T[j_inlet, i] = T_IN
    if is_outlet_top[i]: T[-1, i] = T[-2, i]
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

# Palm rest temperature (front-center, y=40-60mm, x=100-255mm)
palm_mask = (Y >= 0.040) & (Y <= 0.060) & (X >= 0.100) & (X <= 0.255)
T_palm = T[palm_mask].mean() if palm_mask.any() else 0

# Exhaust temperature
T_exhaust = np.mean([T[-2, i] for i in range(NX) if is_outlet_top[i]])

Nu_soc = 0
if soc_m.any() and (T_soc - T_IN) > 0.1:
    c = COMPONENTS[0]
    A_s = (c[2] - c[1]) * 1e-3 * (c[4] - c[3]) * 1e-3
    Nu_soc = c[5] / (A_s * (T_soc - T_IN)) * (c[2] - c[1]) * 1e-3 / K_AIR

print()
print("=" * 62)
print("  RESULTS                       Sim     Real MacBook")
print("=" * 62)
print(f"  Reynolds number     : {RE:.0f}")
print(f"  Max air velocity    : {v_max:.2f} m/s")
print(f"  SoC temperature     : {T_soc:.1f} C     (real: 80-100 C)")
print(f"  Battery max temp    : {T_bat:.1f} C     (real: 38-45 C)")
print(f"  Palm rest temp      : {T_palm:.1f} C     (real: 30-34 C)")
print(f"  Fan exhaust temp    : {T_exhaust:.1f} C     (real: 45-60 C)")
print(f"  Peak T location     : ({x[pi]*1e3:.0f}, {y[pj]*1e3:.0f}) mm")
print(f"  Nusselt on SoC      : {Nu_soc:.1f}")
print("=" * 62)

# === 9-11. FIGURES ======================================================

x_mm, y_mm = x * 1e3, y * 1e3
X_mm, Y_mm = X * 1e3, Y * 1e3

def draw_outlines(ax, color="white", lw=0.7):
    for nm, x0, x1, y0, y1, *_ in COMPONENTS:
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                     lw=lw, edgecolor=color, facecolor="none"))
        lab = nm.replace("_", " ")
        w, h = x1 - x0, y1 - y0
        fs = min(8, max(5.5, min(w, h) / 4.5))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, lab, color=color,
                fontsize=fs, ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4, lw=0))

def mark_io(ax):
    ax.plot([80, 275], [120, 120], lw=2.5, color="deepskyblue", zorder=5)
    ax.annotate("INLET (bottom vents)", xy=(177, 118), fontsize=7,
                color="deepskyblue", ha="center", va="top", fontweight="bold")
    ax.plot([80, 168], [245, 245], lw=3, color="tomato", zorder=5)
    ax.plot([187, 275], [245, 245], lw=3, color="tomato", zorder=5)
    ax.annotate("REAR EXHAUST", xy=(177, 243), fontsize=7,
                color="tomato", ha="center", va="bottom", fontweight="bold")

# Temperature
fig, ax = plt.subplots(figsize=(16, 11))
lv = np.linspace(T.min(), T.max(), 40)
cf = ax.contourf(X_mm, Y_mm, T, levels=lv, cmap="inferno")
fig.colorbar(cf, ax=ax, label="Temperature (°C)", pad=0.01, shrink=0.82)
draw_outlines(ax, "white");  mark_io(ax)
ax.set_title(f"Temperature  |  Re={RE:.0f}  T_SoC={T_soc:.0f}°C  "
             f"T_bat={T_bat:.0f}°C  T_exhaust={T_exhaust:.0f}°C",
             fontsize=10, fontweight="bold")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_xlim(0, 355); ax.set_ylim(0, 245)
plt.tight_layout()
plt.savefig("macbook_temperature.png", dpi=150, bbox_inches="tight")
plt.close()

# Velocity
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
ax.set_title("Velocity — Front-to-Back Airflow", fontsize=11, fontweight="bold")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_xlim(0, 355); ax.set_ylim(0, 245)
plt.tight_layout()
plt.savefig("macbook_velocity.png", dpi=150, bbox_inches="tight")
plt.close()

# Vorticity
fig, ax = plt.subplots(figsize=(16, 11))
wc = 25.0
lv = np.linspace(-wc, wc, 30)
cf = ax.contourf(X_mm, Y_mm, np.clip(omega, -wc, wc), levels=lv, cmap="RdBu_r")
fig.colorbar(cf, ax=ax, label="Vorticity (1/s)", pad=0.01, shrink=0.82)
draw_outlines(ax, "black");  mark_io(ax)
ax.set_title("Vorticity Field", fontsize=11, fontweight="bold")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_xlim(0, 355); ax.set_ylim(0, 245)
plt.tight_layout()
plt.savefig("macbook_vorticity.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n  Figures saved: layout_validation / macbook_temperature / macbook_velocity / macbook_vorticity .png")
