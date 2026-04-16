"""Energy solver vs. 1D analytical conduction.

Configure a tiny domain with no flow, no obstacles, no heat sources;
bottom wall at T_wall, top adiabatic (closed top in our model: there is
no outlet by default, so we add one and check the steady-state inlet ≈
T_wall response when convection is suppressed by zero velocity).

This is an integration test of the matrix assembly: with a uniform
domain, k constant, no source, no convection, T should be uniform =
wall temperature.
"""

from dataclasses import replace

import numpy as np

from cfd import config as cfg
from cfd import energy, geometry
from cfd.flow import FlowField


def _zero_flow(geom):
    ny, nx = geom.ny, geom.nx
    u = np.zeros((ny, nx + 1))
    v = np.zeros((ny + 1, nx))
    p = np.zeros((ny, nx))
    u_c = np.zeros((ny, nx))
    v_c = np.zeros((ny, nx))
    return FlowField(u=u, v=v, p=p, u_cell=u_c, v_cell=v_c,
                     speed_cell=np.zeros((ny, nx)),
                     vorticity_cell=np.zeros((ny, nx)),
                     iterations=0, final_residual=0.0,
                     final_momentum_res=0.0, converged=True,
                     div_history=[], mom_history=[])


def _simple_config():
    """Build a minimal Config without YAML — uniform 0.1 m square, no parts."""
    return cfg.Config(
        domain=cfg.Domain(
            length_m=0.1, height_m=0.1, nx=20, ny=20,
            depth=cfg.DepthClosure(active_zone_m=0.05, battery_zone_m=0.05,
                                    zone_split_y_m=0.0),
        ),
        fluid=cfg.Fluid(rho_kg_m3=1.1, mu_pa_s=1.8e-5,
                        k_air_w_mk=0.025, cp_j_kgk=1000.0,
                        k_chassis_battery_w_mk=0.025),
        bc=cfg.BoundaryConditions(
            wall_temperature_c=30.0,
            inlet_temperature_c=30.0,         # match wall to expect uniform T
            inlet_velocity_m_s=0.0,
            inlet_y_m=0.05,
            inlet_x_range_m=(0.02, 0.08),
            outlet_bands_x_m=((0.04, 0.06),),
        ),
        fan_velocity_m_s=0.0,
        turbulence=cfg.Turbulence(model="none", mixing_length_m=0.0, nu_t_max_ratio=10.0),
        solver=cfg.SolverParams(max_iterations=10, tolerance=1e-3, cfl=0.4,
                                 pressure_relax=0.7, momentum_relax=0.7,
                                 brinkman_penalty_per_s=1e5, log_every=1000),
        components=(),
        validation=cfg.ValidationRanges(
            soc_mean_c=(0, 100), battery_max_c=(0, 100),
            palm_mean_c=(0, 100), exhaust_mean_c=(0, 100),
            palm_x_m=(0.04, 0.06), palm_y_m=(0.04, 0.06),
        ),
    )


def test_uniform_temperature_when_inlet_matches_wall():
    """No source, zero flow, T_in == T_wall → field is exactly T_wall."""
    config = _simple_config()
    geom = geometry.build(config)
    flow = _zero_flow(geom)
    T = energy.solve(config, geom, flow)
    assert np.allclose(T.T, 30.0, atol=1e-6)


def test_temperature_rises_above_wall_when_heat_source():
    """Add a small powered component → T must exceed wall temp somewhere."""
    config = _simple_config()
    comp = cfg.Component(
        name="Heater",
        x_range_m=(0.04, 0.06),
        y_range_m=(0.06, 0.08),
        q_watts=2.0, k_w_mk=200.0,
    )
    config = replace(config, components=(comp,))
    geom = geometry.build(config)
    flow = _zero_flow(geom)
    T = energy.solve(config, geom, flow)
    assert T.T.max() > 30.5
    # Peak temperature should be inside the heater
    peak_j, peak_i = np.unravel_index(T.T.argmax(), T.T.shape)
    assert geom.comp_id[peak_j, peak_i] == 0
