"""Smoke test for the 3D solver on a minimal synthetic domain.

Configures a small 20x20x8 chassis with one inlet on the bottom, one
outlet on the rear, a single heat source, and no components blocking
flow. Checks that the solver runs, produces bounded output, and the
temperature at the heat source exceeds ambient.
"""

import numpy as np

from cfd3d import config3d as cfg, energy3d, flow3d, geometry3d


def _mini_config() -> cfg.Config3D:
    return cfg.Config3D(
        domain=cfg.Domain3D(
            length_x_m=0.1, length_y_m=0.1, length_z_m=0.02,
            nx=20, ny=20, nz=8,
        ),
        fluid=cfg.Fluid3D(rho_kg_m3=1.1, mu_pa_s=1.8e-5,
                          k_air_w_mk=0.025, cp_j_kgk=1000.0),
        bc=cfg.BoundaryConditions3D(
            bottom_wall_temperature_c=25.0,
            ambient_temperature_c=25.0,
            inlet=cfg.InletPatch(
                x_range_m=(0.02, 0.08), y_range_m=(0.03, 0.07),
                velocity_m_s=0.5, temperature_c=25.0,
            ),
            outlets=(
                cfg.OutletPatch(x_range_m=(0.02, 0.08), z_range_m=(0.005, 0.015)),
            ),
        ),
        fan_velocity_m_s=0.0,
        turbulence=cfg.Turbulence3D(model="none", mixing_length_m=0.0, nu_t_max_ratio=10.0),
        solver=cfg.SolverParams3D(
            max_iterations=50, tolerance=1e-3, cfl=0.3,
            pressure_relax=0.8, momentum_relax=0.6,
            brinkman_penalty_per_s=1e5,
            log_every=100,
        ),
        components=(
            cfg.Component3D(
                name="Heater",
                x_range_m=(0.04, 0.06), y_range_m=(0.04, 0.06),
                z_range_m=(0.008, 0.012),
                q_watts=1.0, k_w_mk=100.0, is_fan=False,
            ),
        ),
        validation=cfg.ValidationRanges3D(
            soc_mean_c=(0, 200), battery_max_c=(0, 200),
            palm_mean_c=(0, 200), exhaust_mean_c=(0, 200),
            palm_region_x_m=(0.03, 0.07), palm_region_y_m=(0.01, 0.03),
        ),
    )


def test_3d_solver_runs_and_heats():
    config = _mini_config()
    geom = geometry3d.build(config)
    flow = flow3d.solve(config, geom, log=lambda _m: None)
    assert np.isfinite(flow.speed_cell).all()
    assert flow.speed_cell.max() > 0.1
    T = energy3d.solve(config, geom, flow)
    assert np.isfinite(T.T).all()
    # Heater must raise T above ambient.
    assert T.T.max() > 25.5


def test_3d_geometry_masks_consistent():
    config = _mini_config()
    geom = geometry3d.build(config)
    # Heater occupies known footprint — at least some cells should be flagged.
    assert int(geom.is_solid.sum()) > 0
    # Flow mask excludes the heater.
    assert not (geom.flow_mask & geom.is_solid_flow).any()
    # Inlet patch non-empty.
    assert int(geom.inlet_bottom_mask.sum()) > 0
    # Outlet patch non-empty.
    assert int(geom.outlet_rear_mask.sum()) > 0
