"""Geometry / mask construction tests."""

from pathlib import Path

from cfd import config as cfg
from cfd import geometry


REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "macbook_pro_16.yaml"


def test_mask_shapes_and_counts():
    c = cfg.load(DEFAULT_CONFIG)
    g = geometry.build(c)

    assert g.is_solid.shape == (c.domain.ny, c.domain.nx)
    assert g.is_fan.shape == (c.domain.ny, c.domain.nx)
    assert g.k_field.shape == (c.domain.ny, c.domain.nx)
    assert g.q_volumetric.shape == (c.domain.ny, c.domain.nx)
    # SoC + RAM + ... give a non-trivial solid count
    assert g.is_solid.sum() > 1000
    # At least the two fans are picked up
    assert g.is_fan.sum() > 50
    # Inlet x mask covers more than half the inlet range
    assert g.inlet_face_mask.sum() > 100
    # Outlet bands occupy roughly the same width as inlet
    assert g.outlet_top_mask.sum() > 100


def test_flow_mask_excludes_battery_zone():
    c = cfg.load(DEFAULT_CONFIG)
    g = geometry.build(c)
    # Below j_inlet should never be in flow_mask
    assert not g.flow_mask[: g.j_inlet, :].any()
    # Above j_inlet, fluid cells are in flow_mask
    assert g.flow_mask[g.j_inlet:, :].any()


def test_q_volumetric_only_in_powered_components():
    c = cfg.load(DEFAULT_CONFIG)
    g = geometry.build(c)
    # Q is non-zero strictly inside solid region
    assert (g.q_volumetric > 0).sum() > 0
    assert g.q_volumetric[~g.is_solid].max() == 0.0
