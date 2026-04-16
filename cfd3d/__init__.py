"""3D extension of the MacBook thermal/airflow solver.

The 2D solver approximates the out-of-plane dimension with effective
slab-thickness and conductivity closures. The 3D solver resolves the
chassis depth (z) directly, so fan-duct geometry, vapor-chamber slabs,
and battery-layer stacking become explicit rather than lumped.

Layout mirrors the 2D package:

    config3d.py     YAML loader with 3D bounding-box components
    geometry3d.py   3D staggered MAC grid, masks, source fields
    flow3d.py       3D incompressible Navier–Stokes (Chorin projection)
    energy3d.py     3D steady convection–diffusion for temperature
    visualize3d.py  mid-height and vertical cross-section plots
"""
