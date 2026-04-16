"""
2D Top-Down Forced Convection CFD — MacBook Pro 16-inch
=========================================================
Plan-view: air enters bottom vents → flows across SoC → exits fan exhausts.
Gravity out of plane → no buoyancy.

Approach: Solve the potential-flow stream function on the ENTIRE domain
(treating solid interiors as fluid for the Laplace equation), then mask
velocity to zero inside solids.  This gives smooth, bounded velocities
near solid surfaces — avoiding the numerical blow-up that occurs when
solid cells carry a different psi constant.  The temperature equation
is then solved once as a direct sparse system.
"""

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import splu, spsolve
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# === 1. GRID SETUP =====================================================

L_DOM, H_DOM = 0.355, 0.245
NX, NY = 260, 180
dx, dy = L_DOM / (NX - 1), H_DOM / (NY - 1)
dx2, dy2 = dx**2, dy**2
x = np.linspace(0, L_DOM, NX)
y = np.linspace(0, H_DOM, NY)
X, Y = np.meshgrid(x, y)

V_IN = 0.8;  T_IN = 25.0;  T_WALL = 28.0
X_IN_L, X_IN_R = 0.060, 0.295
PSI_TOTAL = -V_IN * (X_IN_R - X_IN_L)
PSI_MID = PSI_TOTAL / 2.0
H_ACTIVE = 0.235

X_OL_L, X_OL_R = 0.080, 0.175
X_OR_L, X_OR_R = 0.180, 0.275

RHO, MU = 1.109, 1.96e-5
K_AIR, CP = 0.02756, 1007.0
NU = MU / RHO

j_inlet   = int(round(0.120 / dy))
j_chimney = int(round(H_ACTIVE / dy)) + 1

def ix(mm): return int(round(mm * 1e-3 / dx))
i_ol_l, i_ol_r = ix(80), ix(175)
i_or_l, i_or_r = ix(180), ix(275)

# === 2. SOLID MASK & COMPONENT PROPERTIES ==============================

COMPONENTS = [
    ("Vapor Chamber",  100, 255, 148, 215,  0.0, 10000.0),
    ("M-series SoC",   155, 185, 155, 175, 15.0,   150.0),
    ("LPDDR RAM (L)",  130, 155, 158, 172,  2.0,   150.0),
    ("LPDDR RAM (R)",  185, 210, 158, 172,  2.0,   150.0),
    ("SSD Controller", 220, 245, 155, 168,  1.5,   150.0),
    ("Power (MOS)",    215, 235, 175, 188,  1.0,   150.0),
    ("Fan (L)",        100, 160, 195, 235,  0.5,    40.0),
    ("Fan (R)",        195, 255, 195, 235,  0.5,    40.0),
    ("TB4 (L)",         60,  80, 148, 162,  0.8,   150.0),
    ("TB4 (R)",        275, 295, 148, 162,  0.8,   150.0),
    ("Battery 1",       55, 130,  20,  75,  1.5,     5.0),
    ("Battery 2",      135, 210,  20,  95,  1.5,     5.0),
    ("Battery 3",      215, 300,  20,  75,  1.5,     5.0),
    ("Battery 4",       55, 130,  75, 115,  1.0,     5.0),
    ("Battery 5",      215, 300,  75, 115,  1.0,     5.0),
    ("Speaker L",       15,  50,  10, 100,  0.2,    20.0),
    ("Speaker R",      305, 340,  10, 100,  0.2,    20.0),
]
SOC_IDX = 1

is_solid = np.zeros((NY, NX), dtype=bool)
comp_id  = -np.ones((NY, NX), dtype=int)
for ci, (_, x0, x1, y0, y1, *_) in enumerate(COMPONENTS):
    mask = (X >= x0*1e-3) & (X <= x1*1e-3) & (Y >= y0*1e-3) & (Y <= y1*1e-3)
    is_solid |= mask;  comp_id[mask] = ci

# Chimney walls
for j in range(j_chimney, NY):
    for i in range(NX):
        if not ((i_ol_l <= i <= i_ol_r) or (i_or_l <= i <= i_or_r)):
            is_solid[j, i] = True

is_fluid = ~is_solid

# Material arrays
k_arr = np.full((NY, NX), K_AIR)
Q_vol = np.zeros((NY, NX))
for ci, (_, x0, x1, y0, y1, Q_w, k_s) in enumerate(COMPONENTS):
    m = (comp_id == ci);  k_arr[m] = k_s
    if Q_w > 0:
        Q_vol[m] = Q_w / ((x1-x0)*1e-3 * (y1-y0)*1e-3)

# === 3. POTENTIAL FLOW (smooth psi over entire domain) =================
# Solve Laplace (nabla^2 psi = 0) on all interior cells INCLUDING solid
# interiors.  Only chimney walls, enclosure walls, and inlet row are
# Dirichlet.  This produces a smooth psi field without the violent jumps
# that occur when each solid body carries its own constant psi.

print("Solving potential flow ...")

def inlet_psi(xi):
    if xi <= X_IN_L: return 0.0
    if xi >= X_IN_R: return PSI_TOTAL
    return -V_IN * (xi - X_IN_L)

psi_bc = np.zeros((NY, NX))
psi_bc[j_inlet:, -1] = PSI_TOTAL
for i in range(NX): psi_bc[j_inlet, i] = inlet_psi(x[i])
psi_bc[-1, :i_ol_l] = 0.0
psi_bc[-1, i_ol_r+1:i_or_l] = PSI_MID
psi_bc[-1, i_or_r+1:] = PSI_TOTAL
for j in range(j_chimney, NY):
    for i in range(NX):
        if is_solid[j, i]:  # chimney walls
            if x[i] < X_OL_L:          psi_bc[j, i] = 0.0
            elif x[i] > X_OR_R:        psi_bc[j, i] = PSI_TOTAL
            else:                       psi_bc[j, i] = PSI_MID

# Dirichlet: walls, inlet, chimney walls (NOT component solids)
is_dir = np.zeros((NY, NX), dtype=bool)
is_dir[0, :] = True; is_dir[-1, :] = True
is_dir[:, 0] = True; is_dir[:, -1] = True
is_dir[j_inlet, :] = True
for j in range(j_chimney, NY):
    for i in range(NX):
        if is_solid[j, i]: is_dir[j, i] = True

is_outlet_top = np.zeros(NX, dtype=bool)
for i in range(NX):
    if (i_ol_l <= i <= i_ol_r) or (i_or_l <= i <= i_or_r):
        is_outlet_top[i] = True; is_dir[-1, i] = False

# All interior cells that are not Dirichlet → unknowns
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
rhs = np.zeros(N_P)
for j in range(NY):
    for i in range(NX):
        if not psi_uk[j, i]: continue
        k = pidx[j, i]
        A[k, k] = -2/dx2 - 2/dy2
        for dj, di, h2 in [(0,1,dx2),(0,-1,dx2),(1,0,dy2),(-1,0,dy2)]:
            nj, ni = j+dj, i+di
            if 0 <= nj < NY and 0 <= ni < NX:
                nk = pidx[nj, ni]
                if nk >= 0:   A[k, nk] = 1/h2
                elif is_dir[nj, ni]: rhs[k] -= psi_bc[nj, ni]/h2

psi = psi_bc.copy()
psi_uj, psi_ui = np.where(psi_uk)
psi[psi_uj, psi_ui] = splu(A.tocsc()).solve(rhs)
for i in range(NX):
    if is_outlet_top[i]: psi[-1, i] = psi[-2, i]
del A

# Velocity from psi, then zero in solids
u = np.zeros((NY, NX)); v = np.zeros((NY, NX))
u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*dy)
v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*dx)
u[is_solid] = 0; v[is_solid] = 0
# Enforce inlet velocity
for i in range(NX):
    if X_IN_L <= x[i] <= X_IN_R: u[j_inlet, i] = 0; v[j_inlet, i] = V_IN

# Clip extreme velocities at psi-jump boundaries (battery-active interface)
# to keep Pe reasonable for the temperature solve.
speed = np.sqrt(u**2 + v**2)
V_MAX = 2.5 * V_IN   # physical max ~2 m/s
clip = np.where(speed > V_MAX, V_MAX / np.maximum(speed, 1e-30), 1.0)
u *= clip; v *= clip
speed = np.sqrt(u**2 + v**2)
print(f"  Vmax (clipped) = {speed.max():.3f} m/s")

# Vorticity (post-hoc for visualisation)
omega = np.zeros((NY, NX))
omega[1:-1, 1:-1] = ((v[1:-1, 2:]-v[1:-1, :-2])/(2*dx) -
                      (u[2:, 1:-1]-u[:-2, 1:-1])/(2*dy))
omega[is_solid] = 0

# === 4. TEMPERATURE (direct sparse solve) ==============================

print("Building temperature matrix ...")

# Face conductivities
kx_f = np.zeros((NY, NX))
kx_f[:, :-1] = 2*k_arr[:, :-1]*k_arr[:, 1:]/(k_arr[:, :-1]+k_arr[:, 1:])
ky_f = np.zeros((NY, NX))
ky_f[:-1, :] = 2*k_arr[:-1, :]*k_arr[1:, :]/(k_arr[:-1, :]+k_arr[1:, :])

De = kx_f[1:-1,1:-1].copy()/dx2; De[:,-1]=0         # right: Neumann
Dw = np.zeros((NY-2,NX-2)); Dw[:,1:]=kx_f[1:-1,1:-2]/dx2  # left: Neumann (Dw[:,0]=0)
Dn = ky_f[1:-1,1:-1].copy()/dy2; Dn[-1,:]=0         # top: Neumann (adiabatic)
Ds = np.zeros((NY-2,NX-2))
Ds[1:,:]=ky_f[1:-2,1:-1]/dy2; Ds[0,:]=k_arr[1,1:-1]/dy2  # bottom: Dirichlet T_WALL

n_x,n_y = NX-2,NY-2; n_T=n_x*n_y
jj,ii = np.mgrid[0:n_y,0:n_x]; kk=jj*n_x+ii

# Diffusion part
rd,cd,vd=[],[],[]
rd.append(kk.ravel()); cd.append(kk.ravel()); vd.append(-(De+Dw+Dn+Ds).ravel())
for msk,off,D in [((De>0),1,De),((Dw>0),-1,Dw),
                   ((jj<n_y-1),n_x,Dn),((jj>0),-n_x,Ds)]:
    m=msk.ravel(); rd.append(kk.ravel()[m]); cd.append((kk+off).ravel()[m]); vd.append(D.ravel()[m])
A_diff = coo_matrix((np.concatenate(vd),(np.concatenate(rd),np.concatenate(cd))),
                     shape=(n_T,n_T)).tocsc()

rhs_T = -Q_vol[1:-1,1:-1].ravel().copy()
rhs_T[:n_x] -= Ds[0,:].ravel() * T_WALL

# Pin inlet vent cells
inlet_mask = np.zeros((n_y,n_x),dtype=bool)
j_ii = j_inlet-1
if 0<=j_ii<n_y:
    for ic in range(n_x):
        if X_IN_L<=x[ic+1]<=X_IN_R: inlet_mask[j_ii,ic]=True
A_diff = A_diff.tolil()
for idx in np.where(inlet_mask.ravel())[0]:
    A_diff[idx,:]=0; A_diff[idx,idx]=1.0; rhs_T[idx]=T_IN
A_diff = A_diff.tocsc()

# Convection (upwind)
uc=u[1:-1,1:-1].copy(); vc=v[1:-1,1:-1].copy()
uc[inlet_mask]=0; vc[inlet_mask]=0
Fw=RHO*CP*np.maximum(uc,0)/dx; Fe=RHO*CP*np.maximum(-uc,0)/dx
Fs=RHO*CP*np.maximum(vc,0)/dy; Fn=RHO*CP*np.maximum(-vc,0)/dy

# Fp must ONLY sum contributions that have matching off-diagonals.
# At Neumann boundaries (top/left/right), off-diags are missing,
# so including them in Fp creates a spurious sink → T collapse.
offdiag_specs = [((ii<n_x-1)&(Fe>0),1,Fe),((ii>0)&(Fw>0),-1,Fw),
                 ((jj<n_y-1)&(Fn>0),n_x,Fn),((jj>0)&(Fs>0),-n_x,Fs)]

Fp = np.zeros((n_y, n_x))
for msk,_,Fc in offdiag_specs:
    Fp += np.where(msk, Fc, 0)

rc,cc,vc_=[],[],[]
rc.append(kk.ravel()); cc.append(kk.ravel()); vc_.append(-Fp.ravel())
for msk,off,Fc in offdiag_specs:
    m=msk.ravel(); rc.append(kk.ravel()[m]); cc.append((kk+off).ravel()[m]); vc_.append(Fc.ravel()[m])
A_conv = coo_matrix((np.concatenate(vc_),(np.concatenate(rc),np.concatenate(cc))),
                     shape=(n_T,n_T)).tocsc()

print("Solving temperature ...")
T_int = spsolve(A_diff + A_conv, rhs_T).reshape(n_y, n_x)
print(f"  T range = [{T_int.min():.2f}, {T_int.max():.2f}] C")

T = np.full((NY,NX), T_IN)
T[1:-1,1:-1] = T_int
T[0,:]=T_WALL; T[:,0]=T[:,1]; T[:,-1]=T[:,-2]
for i in range(NX):
    if X_IN_L<=x[i]<=X_IN_R: T[j_inlet,i]=T_IN
    if is_outlet_top[i]: T[-1,i]=T[-2,i]

# === 5. POST-PROCESSING ================================================

RE = RHO*V_IN*L_DOM/MU
peak_T = T.max()
pj,pi = np.unravel_index(T.argmax(), T.shape)
v_max = np.where(is_fluid, speed, 0).max()

soc=(comp_id==SOC_IDX)
T_soc = T[soc].mean() if soc.any() else 0

Q_out=0
for i in range(NX):
    if is_outlet_top[i]: Q_out += RHO*CP*max(v[-2,i],0)*(T[-2,i]-T_IN)*dx

Nu_soc=0
if soc.any() and T_soc-T_IN>0.1:
    c=COMPONENTS[SOC_IDX]
    A_s=(c[2]-c[1])*1e-3*(c[4]-c[3])*1e-3
    Nu_soc = c[5]/(A_s*(T_soc-T_IN)) * (c[2]-c[1])*1e-3 / K_AIR

print()
print("="*64)
print("  RESULTS")
print("="*64)
print(f"  Reynolds number            : {RE:.0f}")
print(f"  Max air velocity           : {v_max:.4f} m/s")
print(f"  Avg Nusselt on SoC         : {Nu_soc:.1f}")
print(f"  Peak temperature           : {peak_T:.2f} C  at ({x[pi]*1e3:.1f}, {y[pj]*1e3:.1f}) mm")
print(f"  SoC avg temperature        : {T_soc:.2f} C")
print(f"  Heat via outlets           : {Q_out:.2f} W/m")
print("="*64)

# === 6. VISUALISATION ===================================================

x_mm,y_mm=x*1e3,y*1e3; X_mm,Y_mm=X*1e3,Y*1e3
fig,axes=plt.subplots(3,1,figsize=(20,14),constrained_layout=True)

def draw_outlines(ax,color="white",lw=0.8):
    for nm,x0,x1,y0,y1,*_ in COMPONENTS:
        ax.add_patch(patches.Rectangle((x0,y0),x1-x0,y1-y0,
                     lw=lw,edgecolor=color,facecolor="none"))
        ax.text((x0+x1)/2,(y0+y1)/2,nm,color=color,
                fontsize=4,ha="center",va="center",fontweight="bold")

def mark_io(ax):
    ax.plot([60,295],[120,120],lw=2.5,color="deepskyblue",zorder=5)
    ax.annotate("INLET  v=0.8 m/s",xy=(177,118),fontsize=7,color="deepskyblue",
                ha="center",va="top",fontweight="bold")
    ax.plot([80,175],[235,235],lw=2.5,color="tomato",zorder=5)
    ax.plot([180,275],[235,235],lw=2.5,color="tomato",zorder=5)
    ax.annotate("OUT-L",xy=(127,237),fontsize=6,color="tomato",ha="center",fontweight="bold")
    ax.annotate("OUT-R",xy=(227,237),fontsize=6,color="tomato",ha="center",fontweight="bold")

ax=axes[0]
lv=np.linspace(T.min(),T.max(),40)
cf=ax.contourf(X_mm,Y_mm,T,levels=lv,cmap="inferno")
fig.colorbar(cf,ax=ax,label="Temperature (°C)",pad=0.01)
draw_outlines(ax,"white"); mark_io(ax)
ax.set_title("Temperature Field"); ax.set_ylabel("y (mm)")
ax.set_xlim(0,L_DOM*1e3); ax.set_ylim(0,H_DOM*1e3)

ax=axes[1]
cf=ax.contourf(X_mm,Y_mm,speed,30,cmap="viridis")
fig.colorbar(cf,ax=ax,label="Speed (m/s)",pad=0.01)
ax.streamplot(x_mm,y_mm,u,v,color="white",linewidth=0.8,density=2,arrowsize=0.8)
draw_outlines(ax,"cyan"); mark_io(ax)
ax.set_title("Velocity Streamlines"); ax.set_ylabel("y (mm)")
ax.set_xlim(0,L_DOM*1e3); ax.set_ylim(0,H_DOM*1e3)

ax=axes[2]
wc=20.0; lv=np.linspace(-wc,wc,30)
cf=ax.contourf(X_mm,Y_mm,np.clip(omega,-wc,wc),levels=lv,cmap="RdBu_r")
fig.colorbar(cf,ax=ax,label="Vorticity (1/s)",pad=0.01)
draw_outlines(ax,"black"); mark_io(ax)
ax.set_title("Vorticity Field"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_xlim(0,L_DOM*1e3); ax.set_ylim(0,H_DOM*1e3)

fig.suptitle(
    f"MacBook Pro Forced Convection  |  Re={RE:.0f}  Vmax={v_max:.2f} m/s"
    f"  T_peak={peak_T:.1f}°C  T_SoC={T_soc:.1f}°C",
    fontsize=12,fontweight="bold")
plt.savefig("macbook_pro_forced_convection.png",dpi=150,bbox_inches="tight")
print(f"\n  Figure saved: macbook_pro_forced_convection.png")
plt.close()
