"""
MacBook Pro 16" — 2D Top-Down Forced Convection CFD (v2)
=========================================================
Non-overlapping components, fan forcing, k_field conduction,
potential-flow stream function + direct temperature solve.
"""

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import splu, spsolve
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# === 1. DOMAIN & GRID ===================================================

L, H = 0.355, 0.245          # m
NX, NY = 260, 180
dx, dy = L/(NX-1), H/(NY-1)
dx2, dy2 = dx**2, dy**2
x = np.linspace(0, L, NX)
y = np.linspace(0, H, NY)
X, Y = np.meshgrid(x, y)

V_IN    = 1.2                # inlet velocity [m/s]
T_IN    = 25.0               # inlet temp [C]
T_WALL  = 28.0               # bottom chassis [C]
DEPTH   = 0.130              # effective depth [m] (accounts for 3D z-spreading)

X_IN_L, X_IN_R = 0.080, 0.275
Y_OUT_B, Y_OUT_T = 0.195, 0.230

RHO, MU    = 1.109, 1.96e-5
K_AIR, CP  = 0.02756, 1007.0
NU = MU / RHO

j_inlet = int(round(0.115 / dy))
j_ob    = int(round(Y_OUT_B / dy))
j_ot    = int(round(Y_OUT_T / dy))

# === 2. COMPONENT DEFINITIONS ==========================================

COMPONENTS = [
    # (name,       x0,  x1,  y0,  y1,   Q,    k)     all mm
    ("SoC",        162, 192, 158, 178, 15.0,  150),
    ("RAM_L",      138, 160, 162, 175,  2.0,  150),
    ("RAM_R",      194, 216, 162, 175,  2.0,  150),
    ("VC_L",       108, 160, 180, 218,  0.0, 2000),
    ("VC_C",       162, 194, 178, 200,  0.0, 2000),
    ("VC_R",       196, 248, 180, 218,  0.0, 2000),
    ("Fan_L",      108, 136, 155, 178,  0.5,   50),
    ("Fan_R",      218, 248, 155, 178,  0.5,   50),
    ("SSD_L",      138, 175, 222, 236,  1.5,  150),
    ("SSD_R",      194, 231, 222, 236,  1.5,  150),
    ("TB_L",        18,  38, 158, 172,  0.8,  150),
    ("TB_R",       318, 338, 158, 172,  0.8,  150),
    ("PowerCtrl",  268, 298, 162, 178,  1.0,  150),
    ("Bat_A1",      42, 108,  75, 112,  1.2,    5),
    ("Bat_A2",      42, 108,  20,  73,  1.2,    5),
    ("Bat_B1",     118, 198,  25, 112,  2.0,    5),
    ("Bat_B2",     200, 244,  25, 112,  2.0,    5),
    ("Bat_C1",     246, 315,  75, 112,  1.2,    5),
    ("Bat_C2",     246, 315,  20,  73,  1.2,    5),
    ("Speaker_L",    8,  30,  12, 108,  0.2,   10),
    ("Speaker_R",  326, 348,  12, 108,  0.2,   10),
]

# === 3. OVERLAP CHECK ===================================================

print("Checking for overlapping components ...")
for i in range(len(COMPONENTS)):
    n1, x0a, x1a, y0a, y1a, *_ = COMPONENTS[i]
    for j in range(i+1, len(COMPONENTS)):
        n2, x0b, x1b, y0b, y1b, *_ = COMPONENTS[j]
        if x0a < x1b and x1a > x0b and y0a < y1b and y1a > y0b:
            raise ValueError(f"OVERLAP: {n1} and {n2}")
print("  No overlaps detected.")

# === 4. SOLID MASK & k_field & Q_field ==================================

is_solid = np.zeros((NY, NX), dtype=bool)
is_fan   = np.zeros((NY, NX), dtype=bool)
comp_id  = -np.ones((NY, NX), dtype=int)
k_field  = np.full((NY, NX), K_AIR)
Q_field  = np.zeros((NY, NX))

for ci, (name, x0, x1, y0, y1, Q, k) in enumerate(COMPONENTS):
    mask = ((X >= x0*1e-3) & (X <= x1*1e-3) &
            (Y >= y0*1e-3) & (Y <= y1*1e-3))
    is_solid[mask] = True
    comp_id[mask] = ci
    k_field[mask] = k
    if Q > 0:
        area = (x1-x0)*1e-3 * (y1-y0)*1e-3
        Q_field[mask] = Q / (area * DEPTH)
    if name.startswith("Fan"):
        is_fan[mask] = True

is_fluid = ~is_solid
n_solid  = int(is_solid.sum())
n_fluid  = int(is_fluid.sum())
print(f"  Solid cells: {n_solid}   Fluid cells: {n_fluid}")

# === 5. LAYOUT VALIDATION FIGURE ========================================

print("Generating layout validation ...")
cmap20 = plt.cm.tab20(np.linspace(0, 1, len(COMPONENTS)))
fig, ax = plt.subplots(figsize=(15, 10.5))
ax.add_patch(patches.Rectangle((0,0), 355, 245, lw=2,
             edgecolor="black", facecolor="white"))
for ci, (name, x0, x1, y0, y1, Q, k) in enumerate(COMPONENTS):
    c = cmap20[ci]
    ax.add_patch(patches.Rectangle((x0,y0), x1-x0, y1-y0,
                 lw=1, edgecolor=c, facecolor=c, alpha=0.45))
    ax.text((x0+x1)/2, (y0+y1)/2, name, fontsize=5, ha="center",
            va="center", fontweight="bold", color="black")
ax.plot([80,275],[115,115], lw=2, color="blue", label="Inlet")
ax.plot([0,0],[195,230], lw=2, color="red")
ax.plot([355,355],[195,230], lw=2, color="red", label="Outlets")
ax.axhline(115, color="gray", ls="--", lw=0.5)
ax.set_xlim(-2, 357); ax.set_ylim(-2, 247)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_title("Component Layout — Overlap-Free Validation", fontsize=12)
ax.legend(loc="upper right", fontsize=8)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("layout_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: layout_validation.png")

# === 6. BOUNDARY CONDITIONS & POTENTIAL FLOW ============================

print("Solving potential flow ...")

PSI_TOTAL = -V_IN * (X_IN_R - X_IN_L)
PSI_HALF  = PSI_TOTAL / 2.0

def inlet_psi(xi):
    if xi <= X_IN_L: return 0.0
    if xi >= X_IN_R: return PSI_TOTAL
    return -V_IN * (xi - X_IN_L)

psi_bc = np.zeros((NY, NX))
psi_bc[-1, :] = PSI_HALF                           # top wall
for j in range(NY):                                  # left wall
    if y[j] > Y_OUT_T: psi_bc[j, 0] = PSI_HALF
for j in range(NY):                                  # right wall
    yj = y[j]
    if   yj < 0.115:    psi_bc[j, -1] = 0.0
    elif yj < Y_OUT_B:  psi_bc[j, -1] = PSI_TOTAL
    elif yj <= Y_OUT_T: psi_bc[j, -1] = PSI_TOTAL   # overridden by Neumann
    else:                psi_bc[j, -1] = PSI_HALF
for i in range(NX):                                  # inlet row
    psi_bc[j_inlet, i] = inlet_psi(x[i])

# Dirichlet mask
is_dir = np.zeros((NY, NX), dtype=bool)
is_dir[0, :] = True;  is_dir[-1, :] = True
is_dir[:, 0] = True;  is_dir[:, -1] = True
is_dir[j_inlet, :] = True

# Outlet Neumann: remove from Dirichlet
is_out_L = np.zeros(NY, dtype=bool)
is_out_R = np.zeros(NY, dtype=bool)
for j in range(NY):
    if j_ob <= j <= j_ot:
        is_out_L[j] = True;  is_dir[j, 0]  = False
        is_out_R[j] = True;  is_dir[j, -1] = False

# Poisson unknowns — solve through EVERYTHING (including solid interiors)
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

A = lil_matrix((N_P, N_P))
rhs_p = np.zeros(N_P)
for j in range(NY):
    for i in range(NX):
        if not psi_uk[j, i]: continue
        k = pidx[j, i]
        A[k, k] = -2/dx2 - 2/dy2
        for dj, di, h2 in [(0,1,dx2),(0,-1,dx2),(1,0,dy2),(-1,0,dy2)]:
            nj, ni = j+dj, i+di
            if 0 <= nj < NY and 0 <= ni < NX:
                nk = pidx[nj, ni]
                if nk >= 0:         A[k, nk] = 1/h2
                elif is_dir[nj, ni]: rhs_p[k] -= psi_bc[nj, ni]/h2

psi = psi_bc.copy()
uj, ui = np.where(psi_uk)
psi[uj, ui] = splu(A.tocsc()).solve(rhs_p)
del A
for j in range(NY):
    if is_out_L[j]: psi[j, 0] = psi[j, 1]
    if is_out_R[j]: psi[j, -1] = psi[j, -2]

# Velocity + clip
u = np.zeros((NY, NX));  v = np.zeros((NY, NX))
u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*dy)
v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*dx)
u[is_solid] = 0;  v[is_solid] = 0
for i in range(NX):
    if X_IN_L <= x[i] <= X_IN_R: u[j_inlet, i] = 0; v[j_inlet, i] = V_IN

# Fan forcing: fans push air upward
v[is_fan] = 0.8;  u[is_fan] = 0

speed = np.sqrt(u**2 + v**2)
V_CLIP = 4.0 * V_IN
clip = np.where(speed > V_CLIP, V_CLIP / np.maximum(speed, 1e-30), 1.0)
u *= clip;  v *= clip
speed = np.sqrt(u**2 + v**2)
print(f"  Vmax (clipped) = {speed.max():.2f} m/s")

# Vorticity (post-hoc)
omega = np.zeros((NY, NX))
omega[1:-1, 1:-1] = ((v[1:-1, 2:]-v[1:-1, :-2])/(2*dx) -
                      (u[2:, 1:-1]-u[:-2, 1:-1])/(2*dy))
omega[is_solid & ~is_fan] = 0

# === 7. TEMPERATURE SOLVE ===============================================

print("Building & solving temperature ...")

# Face conductivities using k_field (harmonic mean)
kx_f = np.zeros((NY, NX))
kx_f[:, :-1] = 2*k_field[:, :-1]*k_field[:, 1:] / (k_field[:, :-1]+k_field[:, 1:])
ky_f = np.zeros((NY, NX))
ky_f[:-1, :] = 2*k_field[:-1, :]*k_field[1:, :] / (k_field[:-1, :]+k_field[1:, :])

De = kx_f[1:-1, 1:-1].copy()/dx2;  De[:, -1] = 0
Dw = np.zeros((NY-2, NX-2));       Dw[:, 1:] = kx_f[1:-1, 1:-2]/dx2
Dn = ky_f[1:-1, 1:-1].copy()/dy2;  Dn[-1, :] = 0   # top Neumann
Ds = np.zeros((NY-2, NX-2))
Ds[1:, :] = ky_f[1:-2, 1:-1]/dy2
Ds[0, :]  = k_field[1, 1:-1]/dy2                     # bottom Dirichlet

n_x, n_y = NX-2, NY-2;  n_T = n_x*n_y
jj, ii = np.mgrid[0:n_y, 0:n_x];  kk = jj*n_x + ii

# Diffusion matrix
rd, cd, vd = [], [], []
rd.append(kk.ravel()); cd.append(kk.ravel()); vd.append(-(De+Dw+Dn+Ds).ravel())
for msk, off, D in [((De>0),1,De), ((Dw>0),-1,Dw),
                     ((jj<n_y-1),n_x,Dn), ((jj>0),-n_x,Ds)]:
    m = msk.ravel()
    rd.append(kk.ravel()[m]); cd.append((kk+off).ravel()[m]); vd.append(D.ravel()[m])
A_diff = coo_matrix((np.concatenate(vd), (np.concatenate(rd), np.concatenate(cd))),
                     shape=(n_T, n_T)).tocsc()

# RHS: heat sources (ALL cells) + bottom wall Dirichlet
rhs_T = -Q_field[1:-1, 1:-1].ravel().copy()
rhs_T[:n_x] -= Ds[0, :].ravel() * T_WALL

# Pin inlet cells
inlet_mask = np.zeros((n_y, n_x), dtype=bool)
j_ii = j_inlet - 1
if 0 <= j_ii < n_y:
    for ic in range(n_x):
        if X_IN_L <= x[ic+1] <= X_IN_R:
            inlet_mask[j_ii, ic] = True
A_diff = A_diff.tolil()
for idx in np.where(inlet_mask.ravel())[0]:
    A_diff[idx, :] = 0;  A_diff[idx, idx] = 1.0;  rhs_T[idx] = T_IN
A_diff = A_diff.tocsc()

# Convection (upwind) — corrected Fp: only sum existing off-diagonals
uc = u[1:-1, 1:-1].copy();  vc = v[1:-1, 1:-1].copy()
uc[inlet_mask] = 0;  vc[inlet_mask] = 0
Fw = RHO*CP*np.maximum(uc, 0)/dx;  Fe = RHO*CP*np.maximum(-uc, 0)/dx
Fs = RHO*CP*np.maximum(vc, 0)/dy;  Fn = RHO*CP*np.maximum(-vc, 0)/dy

offdiag_specs = [((ii<n_x-1)&(Fe>0), 1, Fe), ((ii>0)&(Fw>0), -1, Fw),
                 ((jj<n_y-1)&(Fn>0), n_x, Fn), ((jj>0)&(Fs>0), -n_x, Fs)]
Fp = np.zeros((n_y, n_x))
for msk, _, Fc in offdiag_specs:
    Fp += np.where(msk, Fc, 0)

rc, cc, vc_ = [], [], []
rc.append(kk.ravel()); cc.append(kk.ravel()); vc_.append(-Fp.ravel())
for msk, off, Fc in offdiag_specs:
    m = msk.ravel()
    rc.append(kk.ravel()[m]); cc.append((kk+off).ravel()[m]); vc_.append(Fc.ravel()[m])
A_conv = coo_matrix((np.concatenate(vc_), (np.concatenate(rc), np.concatenate(cc))),
                     shape=(n_T, n_T)).tocsc()

T_int = spsolve(A_diff + A_conv, rhs_T).reshape(n_y, n_x)

T = np.full((NY, NX), T_IN)
T[1:-1, 1:-1] = T_int
T[0, :] = T_WALL;  T[:, 0] = T[:, 1];  T[:, -1] = T[:, -2]
for i in range(NX):
    if X_IN_L <= x[i] <= X_IN_R: T[j_inlet, i] = T_IN
for j in range(NY):
    if is_out_L[j]: T[j, 0] = T[j, 1]
    if is_out_R[j]: T[j, -1] = T[j, -2]
T = np.clip(T, T_IN - 1, None)   # clip minor boundary artifacts

print(f"  T range = [{T.min():.2f}, {T.max():.2f}] C")

# === 8. POST-PROCESSING =================================================

RE = RHO * V_IN * L / MU
v_max = np.where(is_fluid | is_fan, speed, 0).max()
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
    A_s = (c[2]-c[1])*1e-3 * (c[4]-c[3])*1e-3
    Nu_soc = c[5] / (A_s*(T_soc-T_IN)) * (c[2]-c[1])*1e-3 / K_AIR

print()
print("=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"  Reynolds number     : {RE:.0f}")
print(f"  Max air velocity    : {v_max:.3f} m/s")
print(f"  Peak temperature    : {peak_T:.2f} C  at ({x[pi]*1e3:.1f}, {y[pj]*1e3:.1f}) mm")
print(f"  SoC avg temperature : {T_soc:.2f} C")
print(f"  Battery max temp    : {T_bat:.2f} C")
print(f"  Nusselt on SoC      : {Nu_soc:.1f}")
print("=" * 60)

# === 9. TEMPERATURE FIGURE ==============================================

x_mm, y_mm = x*1e3, y*1e3
X_mm, Y_mm = X*1e3, Y*1e3

def draw_outlines(ax, color="white", lw=0.7):
    for nm, x0, x1, y0, y1, *_ in COMPONENTS:
        ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0,
                     lw=lw, edgecolor=color, facecolor="none"))
        label = nm.replace("_", "\n")
        ax.text((x0+x1)/2, (y0+y1)/2, label, color=color,
                fontsize=4, ha="center", va="center", fontweight="bold")

def mark_io(ax):
    ax.plot([80,275],[115,115], lw=2.5, color="deepskyblue", zorder=5)
    ax.annotate("INLET v=1.2m/s", xy=(177,113), fontsize=7,
                color="deepskyblue", ha="center", va="top", fontweight="bold")
    ax.plot([0,0],[195,230], lw=2.5, color="tomato", zorder=5)
    ax.plot([355,355],[195,230], lw=2.5, color="tomato", zorder=5)
    ax.annotate("OUT", xy=(4,212), fontsize=6, color="tomato", fontweight="bold")
    ax.annotate("OUT", xy=(351,212), fontsize=6, color="tomato",
                ha="right", fontweight="bold")

fig, ax = plt.subplots(figsize=(16, 11))
lv = np.linspace(T.min(), T.max(), 40)
cf = ax.contourf(X_mm, Y_mm, T, levels=lv, cmap="inferno")
fig.colorbar(cf, ax=ax, label="Temperature (°C)", pad=0.01, shrink=0.8)
draw_outlines(ax, "white"); mark_io(ax)
ax.set_title(f"Temperature  |  Re={RE:.0f}  Vmax={v_max:.1f}m/s  "
             f"T_peak={peak_T:.1f}°C  T_SoC={T_soc:.1f}°C  T_bat={T_bat:.1f}°C",
             fontsize=10, fontweight="bold")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_xlim(0, 355); ax.set_ylim(0, 245)
plt.tight_layout()
plt.savefig("macbook_temperature.png", dpi=150, bbox_inches="tight")
plt.close()

# === 10. VELOCITY FIGURE ================================================

fig, ax = plt.subplots(figsize=(16, 11))
cf = ax.contourf(X_mm, Y_mm, speed, 30, cmap="viridis")
fig.colorbar(cf, ax=ax, label="Speed (m/s)", pad=0.01, shrink=0.8)
ax.streamplot(x_mm, y_mm, u, v, color="white", linewidth=0.8,
              density=2, arrowsize=0.8)
# Mark fan regions with arrows
for nm, x0, x1, y0, y1, *_ in COMPONENTS:
    if nm.startswith("Fan"):
        cx, cy = (x0+x1)/2, (y0+y1)/2
        ax.annotate("", xy=(cx, y1-2), xytext=(cx, y0+2),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=2))
        ax.text(cx, cy, "FAN", color="yellow", fontsize=6,
                ha="center", va="center", fontweight="bold")
draw_outlines(ax, "cyan"); mark_io(ax)
ax.set_title("Velocity Streamlines", fontsize=11, fontweight="bold")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_xlim(0, 355); ax.set_ylim(0, 245)
plt.tight_layout()
plt.savefig("macbook_velocity.png", dpi=150, bbox_inches="tight")
plt.close()

# === 11. VORTICITY FIGURE ===============================================

fig, ax = plt.subplots(figsize=(16, 11))
wc = 25.0
lv = np.linspace(-wc, wc, 30)
cf = ax.contourf(X_mm, Y_mm, np.clip(omega, -wc, wc), levels=lv, cmap="RdBu_r")
fig.colorbar(cf, ax=ax, label="Vorticity (1/s)", pad=0.01, shrink=0.8)
draw_outlines(ax, "black"); mark_io(ax)
ax.set_title("Vorticity Field", fontsize=11, fontweight="bold")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_xlim(0, 355); ax.set_ylim(0, 245)
plt.tight_layout()
plt.savefig("macbook_vorticity.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n  Figures saved:")
print("    layout_validation.png")
print("    macbook_temperature.png")
print("    macbook_velocity.png")
print("    macbook_vorticity.png")
