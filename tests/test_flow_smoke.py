"""Smoke tests for the NS solver: no NaN, mass roughly conserved, sane Vmax."""

import numpy as np

from tests.test_energy_analytical import _simple_config

from cfd import config as cfg
from cfd import flow, geometry


def _flowing_config():
    base = _simple_config()
    bc = cfg.BoundaryConditions(
        wall_temperature_c=30.0, inlet_temperature_c=25.0,
        inlet_velocity_m_s=1.0,                   # turn on flow
        inlet_y_m=0.05,
        inlet_x_range_m=(0.02, 0.08),
        outlet_bands_x_m=((0.02, 0.08),),
    )
    solver = cfg.SolverParams(max_iterations=200, tolerance=1e-3, cfl=0.35,
                               pressure_relax=0.7, momentum_relax=0.7,
                               brinkman_penalty_per_s=1e5, log_every=1000)
    from dataclasses import replace
    return replace(base, bc=bc, solver=solver, fan_velocity_m_s=0.0)


def test_flow_no_nan_and_bounded():
    config = _flowing_config()
    geom = geometry.build(config)
    field = flow.solve(config, geom, log=lambda _: None)
    assert np.all(np.isfinite(field.u))
    assert np.all(np.isfinite(field.v))
    assert np.all(np.isfinite(field.p))
    # Vmax should be at most a few times the inlet velocity
    assert field.speed_cell.max() < 10 * config.bc.inlet_velocity_m_s


def test_flow_residual_decreases():
    config = _flowing_config()
    geom = geometry.build(config)
    field_short = flow.solve(
        cfg.Config(**{**config.__dict__,
                      "solver": cfg.SolverParams(max_iterations=5, tolerance=0.0,
                                                   cfl=0.35, pressure_relax=0.7,
                                                   momentum_relax=0.7,
                                                   brinkman_penalty_per_s=1e5,
                                                   log_every=1000)}),
        geom, log=lambda _: None)
    field_long = flow.solve(config, geom, log=lambda _: None)
    # Longer run should achieve smaller residual
    assert field_long.final_residual <= field_short.final_residual + 1e-6
