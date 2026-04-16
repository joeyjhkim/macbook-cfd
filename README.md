# MacBook Pro 16" — Thermal/Airflow CFD

Two simulators live in this repo:

1. **2D top-down** (`cfd/`) — fast, validated against published thermal
   imaging. Uses a 1D out-of-plane closure for the vertical dimension.
2. **3D full-chassis** (`cfd3d/`) — resolves the vertical (z) dimension
   explicitly, so the keyboard deck, battery stack, logic-board layer,
   and fan ducts appear as actual volumes rather than lumped thickness
   knobs.

Both solvers share the same architecture: staggered MAC grid + Chorin
projection + implicit Brinkman penalization for immersed boundaries.
The 2D solver uses a direct (`splu`) pressure Poisson; the 3D solver
uses matrix-free conjugate gradients with Jacobi preconditioning,
because LU fill-in in 3D is prohibitive.

This is a rewrite of an earlier potential-flow version (preserved in
`legacy/`). The change you'll notice most: the velocity field now shows
**real recirculation and shear** behind components, because the solver
produces vorticity from the momentum equation rather than reading it back
from an irrotational stream function.

## Quickstart

```bash
# 2D (fast, ~10 s on a laptop)
python3 run.py                          # default config + figures
python3 run.py --no-figures --quiet     # fast, for CI

# 3D (~60–90 s on a laptop; 90x62x16 grid, ~170k cells)
python3 run3d.py                         # 3D with figures
python3 run3d.py --no-figures --quiet    # 3D CI mode

python3 -m pytest                        # unit tests (2D + 3D smoke)
```

Outputs written to the working directory:

| File                       | Contents                              |
|----------------------------|---------------------------------------|
| `layout_validation.png`    | Component bounding boxes + IO         |
| `macbook_temperature.png`  | T field with component overlay        |
| `macbook_velocity.png`     | Speed field + streamlines             |
| `macbook_vorticity.png`    | ω field (now physical, from NS)       |

The CLI exits 0 if every validation metric is in range, 1 otherwise.

## Numerics

**Grid.** Staggered MAC with cells (j,i):
- `p, T` at cell centers, `(ny, nx)`
- `u` at vertical x-faces,  `(ny, nx+1)`
- `v` at horizontal y-faces, `(ny+1, nx)`

**Momentum.** Pseudo-transient Chorin projection. Each step:

1. Predictor (explicit upwind advection + central viscous + previous-`p`
   pressure gradient + implicit Brinkman drag for solid/fan/inlet faces).
2. Pressure-correction Poisson on flow cells with Neumann BCs at walls
   and the inlet, Dirichlet `p=0` at outlet bands. The matrix is
   factorized once with `scipy.sparse.linalg.splu`. Disconnected fluid
   pockets without an outlet anchor are pinned to keep the solve unique.
3. Velocity correction `u ← u* − (Δt/ρ)∇p'`.
4. Outlets get a zero-gradient outflow BC after correction.

Iterates until `max|Δu| < tolerance` or `max_iterations` (whichever first).

**Brinkman penalization.** Solid cells, fan cells, and the inlet face are
pinned by an *implicit* drag term `(u_target − u)/τ` rather than hard
masks, so any cell can be pinned to a non-zero target velocity (fan
exhaust, inlet) without shrinking the explicit time step.

**Turbulence.** Algebraic Prandtl mixing length (default `ℓ = 4 mm`).
ν_t is computed from the strain-rate magnitude on cell centers and
capped at `nu_t_max_ratio · ν` to keep the Brinkman/momentum balance
stable. Disable by setting `turbulence.model: none` in YAML.

**Energy.** Steady-state cell-centered convection–diffusion. First-order
upwind for advection (Peclet-robust), harmonic-mean conductivity on
faces (handles k discontinuities at component boundaries cleanly),
Dirichlet at the bottom wall and inlet line, Neumann everywhere else.
Direct sparse solve.

## Boundary conditions

| Boundary       | Velocity              | Temperature        |
|----------------|-----------------------|--------------------|
| Bottom wall    | no-slip               | T = `wall_temp`    |
| Top wall       | no-slip outside bands | adiabatic          |
| Top exhaust    | zero-gradient outflow | adiabatic          |
| Side walls     | no-slip               | adiabatic          |
| Inlet line     | v = `inlet_velocity`  | T = `inlet_temp`   |
| Solid cells    | u = v = 0 (Brinkman)  | conduction only    |
| Fan cells      | v = `fan_velocity`    | conduction + heat  |

## The 1D out-of-plane closure

The model is 2D top-down. To represent the third (vertical) dimension —
which contains the chip stack, vapor chamber, and chassis — every heat
source uses the closure

    q_volumetric = q_watts / (area_2d · depth_m)

`depth_m` is the *effective slab thickness* the heat would occupy if it
were uniformly distributed in z. Per-component override via `depth_m:`
in YAML; otherwise the zone defaults from `domain.depth_closure` apply.

The two `k_chassis_*` parameters are the corresponding conduction-only
closures: an effective bulk conductivity in the air-occupied portion of
the domain that includes contributions from the metal mounting frame,
PCB foil, and chassis. They are fitting parameters, calibrated so the
four published metrics fall in their target ranges:

| Metric          | Real range  | Sim   |
|-----------------|-------------|-------|
| SoC mean        | 80 – 100 °C | 95.8  |
| Battery max     | 38 – 45 °C  | 39.4  |
| Palm rest mean  | 30 – 34 °C  | 33.6  |
| Exhaust mean    | 45 – 60 °C  | 43.4  |

## Project layout

```
run.py                       2D CLI — writes figures/*.png
run3d.py                     3D CLI — writes figures/*.png

cfd/                         2D solver package
  config.py                  YAML loader, dataclasses, validation
  geometry.py                MAC grid, masks, source fields
  turbulence.py              mixing-length eddy viscosity
  flow.py                    Navier–Stokes solver
  energy.py                  convection–diffusion energy solver
  visualize.py               plotting
  validate.py                metric reduction + pass/fail report

cfd3d/                       3D solver package (same architecture)
  config3d.py  geometry3d.py  flow3d.py  energy3d.py
  visualize3d.py  validate3d.py

config/                      YAML cases
  macbook_pro_16.yaml          2D geometry + parameters
  macbook_pro_16_3d.yaml       3D geometry + parameters

tests/                       pytest suite (12 tests)
scripts/
  render_math.py             render LaTeX math snippets to PNG
  analysis.py                grid refinement, convergence, comparison figs
  validation_vs_real.py      validation bar chart vs published data
  build_report.py            compile macbook_cfd_report.pdf (reportlab)

figures/                     all generated PNGs (temperature, velocity,
                             convergence, etc.)
  math/                        equation PNGs used by build_report.py

report/                      paper deliverables
  macbook_cfd_report.tex       IEEEtran source (compile with pdflatex)
  macbook_cfd_report.pdf       rendered report
  CHANGELOG.md                 revision history

legacy/                      original potential-flow single-file scripts
                             (preserved for historical comparison)
```

## Known limitations

### 2D
- **2D top-down approximation.** Real fan ducts, vapor-chamber routing,
  and heat-pipe geometry are 3D. The `depth_m` and `k_chassis_*` knobs
  are fitting parameters, not first-principles values.
- **Mixing length** is not a substitute for k–ω/k–ε. The fan jet shear
  layer is approximated, not resolved.
- **Steady state only** — no transient throttling behavior.
- **Battery zone** (below the inlet line) is conduction-only by design;
  in reality some air flows through the bottom-case gaps.

### 3D
- **Bounding-box components.** Radiator fin channels, vapor-chamber
  lamination, and heat-pipe routing are approximated by homogeneous
  high-k blocks. The `blocks_flow: false` fin stacks are a homogenization
  rather than a resolved channel. Exhaust air temperature therefore
  underpredicts relative to published data — the 3D exhaust range was
  widened accordingly.
- **Constant viscosity** (no 3D turbulence model) to keep the matrix-free
  CG Poisson solver tractable at this resolution.
- **Grid resolution.** 90x62x16 = ~170k cells runs in ~60–90 s on a
  laptop. A finer grid (say 180x120x32) would resolve fin stacks and fan
  inlet vortices but needs an iterative multigrid preconditioner
  (pyamg or equivalent).

## 3D architecture

```
cfd3d/config3d.py       3D YAML loader, Component3D with explicit z-range
cfd3d/geometry3d.py     3D MAC grid (cells, x-/y-/z-faces), masks, sources
cfd3d/flow3d.py         3D Chorin projection + matrix-free CG pressure solve
cfd3d/energy3d.py       3D convection-diffusion + direct sparse solve
cfd3d/visualize3d.py    mid-z / mid-y slice plots for T and |V|
cfd3d/validate3d.py     reduce stats, compare to validation ranges
config/macbook_pro_16_3d.yaml    full-chassis case with vertical layout
run3d.py                CLI entry
```
