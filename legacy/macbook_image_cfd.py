"""
MacBook Pro 16" — Image-Based 2D Forced Convection CFD
=======================================================
Step 1: Extract component layout from teardown image
Step 2: Run forced convection CFD with those positions
"""

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import splu, spsolve
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# === 1. LOAD AND ANALYZE IMAGE ==========================================

img = Image.open("mac_components.png")
IMG_W, IMG_H = img.size   # 651 × 457
img_arr = np.array(img)

# MacBook Pro 16-inch physical dimensions
PHYS_W, PHYS_H = 355.0, 245.0   # mm
mm_per_px_x = PHYS_W / IMG_W     # 0.545 mm/px
mm_per_px_y = PHYS_H / IMG_H     # 0.536 mm/px

print(f"Image: {IMG_W}×{IMG_H} px")
print(f"Scale: {mm_per_px_x:.4f} mm/px (x), {mm_per_px_y:.4f} mm/px (y)")

# === 2. EXTRACT COMPONENT BOUNDING BOXES ================================
# Coordinates from visual inspection of the teardown image.
# Image convention: top=hinge (y_mm=245), bottom=front (y_mm=0).
# Convert pixel (px_x, px_y) → mm: x_mm = px_x * scale_x
#                                    y_mm = 245 - px_y * scale_y

def px_to_mm(px_x, px_y):
    return round(px_x * mm_per_px_x), round(PHYS_H - px_y * mm_per_px_y)

# Component positions extracted from the teardown image (mm, y=0 at front)
COMPONENTS = {
    # --- Active-zone logic board ---
    "SoC":            {"x": [163, 192], "y": [168, 194], "Q": 15.0,  "k": 150},
    "RAM_L":          {"x": [138, 163], "y": [172, 191], "Q": 2.0,   "k": 150},
    "RAM_R":          {"x": [192, 217], "y": [172, 191], "Q": 2.0,   "k": 150},
    "VaporChamber_L": {"x": [ 98, 155], "y": [158, 218], "Q": 0.0,   "k": 10000},
    "VaporChamber_R": {"x": [200, 257], "y": [158, 218], "Q": 0.0,   "k": 10000},
    "VaporChamber_C": {"x": [148, 207], "y": [155, 200], "Q": 0.0,   "k": 10000},
    "Fan_L":          {"x": [ 58, 148], "y": [160, 232], "Q": 0.5,   "k": 50},
    "Fan_R":          {"x": [207, 297], "y": [160, 232], "Q": 0.5,   "k": 50},
    "SSD_L":          {"x": [128, 158], "y": [210, 228], "Q": 1.5,   "k": 150},
    "SSD_R":          {"x": [197, 227], "y": [210, 228], "Q": 1.5,   "k": 150},
    "TB_L":           {"x": [ 38,  58], "y": [170, 190], "Q": 0.8,   "k": 150},
    "TB_R":           {"x": [297, 317], "y": [170, 190], "Q": 0.8,   "k": 150},
    "PowerCtrl":      {"x": [270, 295], "y": [173, 192], "Q": 1.0,   "k": 150},
    # --- Battery zone ---
    "Bat_A1":         {"x": [ 35, 112], "y": [ 82, 112], "Q": 1.2,   "k": 5},
    "Bat_A2":         {"x": [ 35, 112], "y": [ 22,  78], "Q": 1.2,   "k": 5},
    "Bat_B1":         {"x": [118, 200], "y": [ 22, 112], "Q": 2.0,   "k": 5},
    "Bat_B2":         {"x": [200, 282], "y": [ 22, 112], "Q": 2.0,   "k": 5},
    "Bat_C1":         {"x": [243, 320], "y": [ 82, 112], "Q": 1.2,   "k": 5},
    "Bat_C2":         {"x": [243, 320], "y": [ 22,  78], "Q": 1.2,   "k": 5},
    "Speaker_L":      {"x": [  8,  30], "y": [ 12,  95], "Q": 0.2,   "k": 10},
    "Speaker_R":      {"x": [325, 347], "y": [ 12,  95], "Q": 0.2,   "k": 10},
}

# === 3. GENERATE VALIDATION FIGURE ======================================

print("Generating layout validation figure ...")

cmap_comp = plt.cm.tab20(np.linspace(0, 1, len(COMPONENTS)))
fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(img_arr, extent=[0, PHYS_W, 0, PHYS_H], aspect="auto", alpha=0.7)

for idx, (name, info) in enumerate(COMPONENTS.items()):
    x0, x1 = info["x"]
    y0, y1 = info["y"]
    color = cmap_comp[idx]
    rect = patches.Rectangle((x0, y0), x1-x0, y1-y0,
                              lw=1.5, edgecolor=color, facecolor=color, alpha=0.25)
    ax.add_patch(rect)
    ax.text((x0+x1)/2, (y0+y1)/2, name.replace("_", "\n"),
            color="white", fontsize=5.5, ha="center", va="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, lw=0))

ax.set_xlim(0, PHYS_W)
ax.set_ylim(0, PHYS_H)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title("Component Layout Extracted from Teardown Image", fontsize=13)
plt.tight_layout()
plt.savefig("macbook_layout_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: macbook_layout_validation.png")

# === 4. BUILD GRID & SOLID MASK =========================================

L_DOM, H_DOM = PHYS_W * 1e-3, PHYS_H * 1e-3
NX, NY = 260, 180
dx, dy = L_DOM/(NX-1), H_DOM/(NY-1)
dx2, dy2 = dx**2, dy**2
x = np.linspace(0, L_DOM, NX)
y = np.linspace(0, H_DOM, NY)
X, Y = np.meshgrid(x, y)

# Flow parameters
# Flow parameters
V_IN = 1.2;  T_IN = 25.0;  T_WALL = 28.0
# 2D simulation uses unit depth; scale Q to get realistic temperatures
# (5 mm chassis → effective Q_2d = Q_3d / depth_ratio)
DEPTH = 0.045   # effective depth [m] tuned for realistic SoC temps (~55-70 C)
X_IN_L, X_IN_R = 0.080, 0.275

# Side outlets (left & right walls)
Y_OUT_BOT, Y_OUT_TOP = 0.190, 0.230
PSI_TOTAL = -V_IN * (X_IN_R - X_IN_L)   # -0.234
PSI_HALF = PSI_TOTAL / 2.0                # -0.117

# Air at 45 C
RHO, MU = 1.109, 1.96e-5
K_AIR, CP = 0.02756, 1007.0
NU = MU / RHO

j_inlet = int(round(0.115 / dy))
j_out_b = int(round(Y_OUT_BOT / dy))
j_out_t = int(round(Y_OUT_TOP / dy))

# Component names for ordered iteration (VC first so others overwrite)
COMP_ORDER = (["VaporChamber_L", "VaporChamber_C", "VaporChamber_R"] +
              [n for n in COMPONENTS if not n.startswith("Vapor")])

is_solid = np.zeros((NY, NX), dtype=bool)
comp_name_map = {}  # name → integer id
comp_id = -np.ones((NY, NX), dtype=int)

for ci, name in enumerate(COMP_ORDER):
    info = COMPONENTS[name]
    comp_name_map[name] = ci
    x0, x1 = [v*1e-3 for v in info["x"]]
    y0, y1 = [v*1e-3 for v in info["y"]]
    mask = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    is_solid |= mask
    comp_id[mask] = ci

is_fluid = ~is_solid

# === 5. HEAT SOURCE MAP =================================================

k_arr = np.full((NY, NX), K_AIR)
Q_vol = np.zeros((NY, NX))

for name in COMP_ORDER:
    info = COMPONENTS[name]
    ci = comp_name_map[name]
    m = (comp_id == ci)
    k_arr[m] = info["k"]
    if info["Q"] > 0:
        x0, x1 = info["x"];  y0, y1 = info["y"]
        area = (x1-x0)*1e-3 * (y1-y0)*1e-3
        Q_vol[m] = info["Q"] / (area * DEPTH)

# === 6. BOUNDARY CONDITIONS & POTENTIAL FLOW ============================

print("Solving potential flow ...")

def inlet_psi(xi):
    if xi <= X_IN_L: return 0.0
    if xi >= X_IN_R: return PSI_TOTAL
    return -V_IN * (xi - X_IN_L)

psi_bc = np.zeros((NY, NX))

# Bottom wall: psi = 0 (already)
# Top wall: psi = PSI_HALF
psi_bc[-1, :] = PSI_HALF

# Left wall (i=0)
for j in range(NY):
    if y[j] > Y_OUT_TOP:
        psi_bc[j, 0] = PSI_HALF

# Right wall (i=NX-1)
for j in range(NY):
    yj = y[j]
    if yj < 0.115:
        psi_bc[j, -1] = 0.0      # battery zone
    elif yj < Y_OUT_BOT:
        psi_bc[j, -1] = PSI_TOTAL  # active zone below outlet
    elif yj <= Y_OUT_TOP:
        psi_bc[j, -1] = PSI_TOTAL  # will be overridden to Neumann
    else:
        psi_bc[j, -1] = PSI_HALF   # above outlet

# Inlet row
for i in range(NX):
    psi_bc[j_inlet, i] = inlet_psi(x[i])

# Dirichlet mask
is_dir = np.zeros((NY, NX), dtype=bool)
is_dir[0, :] = True;  is_dir[-1, :] = True
is_dir[:, 0] = True;  is_dir[:, -1] = True
is_dir[j_inlet, :] = True

# Outlet cells → Neumann, not Dirichlet
is_out_L = np.zeros(NY, dtype=bool)
is_out_R = np.zeros(NY, dtype=bool)
for j in range(NY):
    if j_out_b <= j <= j_out_t:
        is_out_L[j] = True;  is_dir[j, 0] = False
        is_out_R[j] = True;  is_dir[j, -1] = False

# Poisson unknowns (interior, not Dirichlet, not inlet row)
psi_uk = np.zeros((NY, NX), dtype=bool)
psi_uk[1:-1, 1:-1] = True
psi_uk[j_inlet, :] = False
psi_uk &= ~is_dir

pidx = -np.ones((NY, NX), dtype=int)
c = 0
for j in range(NY):
    for i in range(NX):
        if psi_uk[j, i]:
            pidx[j, i] = c; c += 1
N_P = c

A_psi = lil_matrix((N_P, N_P))
rhs_psi = np.zeros(N_P)
for j in range(NY):
    for i in range(NX):
        if not psi_uk[j, i]: continue
        k = pidx[j, i]
        A_psi[k, k] = -2/dx2 - 2/dy2
        for dj, di, h2 in [(0,1,dx2),(0,-1,dx2),(1,0,dy2),(-1,0,dy2)]:
            nj, ni = j+dj, i+di
            if 0 <= nj < NY and 0 <= ni < NX:
                nk = pidx[nj, ni]
                if nk >= 0:     A_psi[k, nk] = 1/h2
                elif is_dir[nj, ni]: rhs_psi[k] -= psi_bc[nj, ni]/h2

psi = psi_bc.copy()
uj, ui = np.where(psi_uk)
psi[uj, ui] = splu(A_psi.tocsc()).solve(rhs_psi)
del A_psi

# Outlet Neumann
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

speed = np.sqrt(u**2 + v**2)
V_CLIP = 3.0 * V_IN
clip = np.where(speed > V_CLIP, V_CLIP / np.maximum(speed, 1e-30), 1.0)
u *= clip;  v *= clip
speed = np.sqrt(u**2 + v**2)
print(f"  Vmax (clipped) = {speed.max():.2f} m/s")

# Vorticity (post-hoc)
omega = np.zeros((NY, NX))
omega[1:-1, 1:-1] = ((v[1:-1, 2:]-v[1:-1, :-2])/(2*dx) -
                      (u[2:, 1:-1]-u[:-2, 1:-1])/(2*dy))
omega[is_solid] = 0

# === 7. TEMPERATURE SOLVE ===============================================

print("Building & solving temperature ...")

# Face conductivities (harmonic mean)
kx_f = np.zeros((NY, NX))
kx_f[:, :-1] = 2*k_arr[:, :-1]*k_arr[:, 1:]/(k_arr[:, :-1]+k_arr[:, 1:])
ky_f = np.zeros((NY, NX))
ky_f[:-1, :] = 2*k_arr[:-1, :]*k_arr[1:, :]/(k_arr[:-1, :]+k_arr[1:, :])

De = kx_f[1:-1, 1:-1].copy()/dx2
Dw = np.zeros((NY-2, NX-2));  Dw[:, 1:] = kx_f[1:-1, 1:-2]/dx2
Dn = ky_f[1:-1, 1:-1].copy()/dy2
Ds = np.zeros((NY-2, NX-2))
Ds[1:, :] = ky_f[1:-2, 1:-1]/dy2;  Ds[0, :] = k_arr[1, 1:-1]/dy2  # bottom Dirichlet

# Neumann walls: zero face diffusion so diagonal doesn't create a sink
De[:, -1] = 0          # right wall adiabatic / outlet
# Dw[:, 0] already 0   # left wall adiabatic / outlet
Dn[-1, :] = 0          # top wall adiabatic

n_x, n_y = NX-2, NY-2;  n_T = n_x*n_y
jj, ii = np.mgrid[0:n_y, 0:n_x];  kk = jj*n_x + ii

# Diffusion matrix
rd, cd, vd = [], [], []
rd.append(kk.ravel()); cd.append(kk.ravel()); vd.append(-(De+Dw+Dn+Ds).ravel())
for msk, off, D in [((De>0),1,De),((Dw>0),-1,Dw),
                     ((jj<n_y-1),n_x,Dn),((jj>0),-n_x,Ds)]:
    m = msk.ravel()
    rd.append(kk.ravel()[m]); cd.append((kk+off).ravel()[m]); vd.append(D.ravel()[m])
A_diff = coo_matrix((np.concatenate(vd), (np.concatenate(rd), np.concatenate(cd))),
                     shape=(n_T, n_T)).tocsc()

rhs_T = -Q_vol[1:-1, 1:-1].ravel().copy()
rhs_T[:n_x] -= Ds[0, :].ravel() * T_WALL

# Pin inlet cells to T_IN
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

# Convection (upwind) — use corrected Fp (only sum existing off-diags)
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

# Clip minor boundary artifacts (outlet cells can undershoot slightly)
T = np.clip(T, T_IN, None)
print(f"  T range = [{T.min():.2f}, {T.max():.2f}] C")

# === 8. POST-PROCESSING =================================================

RE = RHO * V_IN * L_DOM / MU
v_max = np.where(is_fluid, speed, 0).max()
peak_T = T.max()
pj, pi = np.unravel_index(T.argmax(), T.shape)

SOC_CI = comp_name_map["SoC"]
soc_m = (comp_id == SOC_CI)
T_soc = T[soc_m].mean() if soc_m.any() else 0

bat_names = [n for n in COMP_ORDER if n.startswith("Bat")]
bat_m = np.zeros((NY, NX), dtype=bool)
for bn in bat_names:
    bat_m |= (comp_id == comp_name_map[bn])
T_bat_max = T[bat_m].max() if bat_m.any() else 0

Nu_soc = 0
if soc_m.any() and (T_soc - T_IN) > 0.01:
    info = COMPONENTS["SoC"]
    A_s = (info["x"][1]-info["x"][0])*1e-3 * (info["y"][1]-info["y"][0])*1e-3
    h_c = info["Q"] / (A_s * (T_soc - T_IN))
    Nu_soc = h_c * (info["x"][1]-info["x"][0])*1e-3 / K_AIR

print()
print("=" * 64)
print("  RESULTS")
print("=" * 64)
print(f"  Reynolds number       : {RE:.0f}")
print(f"  Max air velocity      : {v_max:.4f} m/s")
print(f"  Peak temperature      : {peak_T:.2f} C  at ({x[pi]*1e3:.1f}, {y[pj]*1e3:.1f}) mm")
print(f"  SoC avg temperature   : {T_soc:.2f} C")
print(f"  Battery max temp      : {T_bat_max:.2f} C")
print(f"  Nusselt on SoC        : {Nu_soc:.1f}")
print("=" * 64)

# === 9-11. FIGURES =======================================================

x_mm, y_mm = x*1e3, y*1e3
X_mm, Y_mm = X*1e3, Y*1e3

def draw_outlines(ax, color="white", lw=0.7):
    for name, info in COMPONENTS.items():
        x0, x1 = info["x"];  y0, y1 = info["y"]
        ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0,
                     lw=lw, edgecolor=color, facecolor="none"))
        label = name.replace("VaporChamber_", "VC-").replace("_", "\n")
        ax.text((x0+x1)/2, (y0+y1)/2, label, color=color,
                fontsize=4, ha="center", va="center", fontweight="bold")

def mark_io(ax):
    ax.plot([80, 275], [115, 115], lw=2.5, color="deepskyblue", zorder=5)
    ax.annotate("INLET v=1.2m/s", xy=(177, 113), fontsize=7,
                color="deepskyblue", ha="center", va="top", fontweight="bold")
    ax.plot([0, 0], [190, 230], lw=2.5, color="tomato", zorder=5)
    ax.plot([355, 355], [190, 230], lw=2.5, color="tomato", zorder=5)
    ax.annotate("OUT", xy=(3, 210), fontsize=6, color="tomato", fontweight="bold")
    ax.annotate("OUT", xy=(348, 210), fontsize=6, color="tomato",
                fontweight="bold", ha="right")

# Figure 2 — Temperature
fig, ax = plt.subplots(figsize=(16, 11))
lv = np.linspace(T.min(), T.max(), 40)
cf = ax.contourf(X_mm, Y_mm, T, levels=lv, cmap="inferno")
fig.colorbar(cf, ax=ax, label="Temperature (°C)", pad=0.01, shrink=0.8)
draw_outlines(ax, "white");  mark_io(ax)
ax.set_title(f"Temperature  |  Re={RE:.0f}  Vmax={v_max:.2f}m/s"
             f"  T_peak={peak_T:.1f}°C  T_SoC={T_soc:.1f}°C",
             fontsize=11, fontweight="bold")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_xlim(0, PHYS_W); ax.set_ylim(0, PHYS_H)
plt.tight_layout()
plt.savefig("macbook_temperature.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 3 — Velocity
fig, ax = plt.subplots(figsize=(16, 11))
cf = ax.contourf(X_mm, Y_mm, speed, 30, cmap="viridis")
fig.colorbar(cf, ax=ax, label="Speed (m/s)", pad=0.01, shrink=0.8)
ax.streamplot(x_mm, y_mm, u, v, color="white", linewidth=0.8,
              density=2.5, arrowsize=0.8)
draw_outlines(ax, "cyan");  mark_io(ax)
ax.set_title("Velocity Streamlines", fontsize=11, fontweight="bold")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_xlim(0, PHYS_W); ax.set_ylim(0, PHYS_H)
plt.tight_layout()
plt.savefig("macbook_velocity.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 4 — Vorticity
fig, ax = plt.subplots(figsize=(16, 11))
wc = 25.0
lv = np.linspace(-wc, wc, 30)
cf = ax.contourf(X_mm, Y_mm, np.clip(omega, -wc, wc), levels=lv, cmap="RdBu_r")
fig.colorbar(cf, ax=ax, label="Vorticity (1/s)", pad=0.01, shrink=0.8)
draw_outlines(ax, "black");  mark_io(ax)
ax.set_title("Vorticity Field", fontsize=11, fontweight="bold")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_xlim(0, PHYS_W); ax.set_ylim(0, PHYS_H)
plt.tight_layout()
plt.savefig("macbook_vorticity.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n  Figures saved:")
print("    macbook_layout_validation.png")
print("    macbook_temperature.png")
print("    macbook_velocity.png")
print("    macbook_vorticity.png")
