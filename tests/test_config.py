"""Config loader: validation and overlap detection."""

from pathlib import Path

import pytest
import yaml

from cfd import config as cfg


REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config" / "macbook_pro_16.yaml"


def test_loads_default_config():
    c = cfg.load(DEFAULT_CONFIG)
    assert c.domain.nx == 260
    assert c.domain.ny == 180
    assert len(c.components) > 10
    assert any(comp.is_fan for comp in c.components)
    assert c.fluid.nu_m2_s > 0


def test_overlap_rejected(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    # Make two components overlap
    raw["components"] = [
        {"name": "A", "x_mm": [10, 30], "y_mm": [130, 150], "q_w": 1, "k_w_mk": 100},
        {"name": "B", "x_mm": [20, 40], "y_mm": [140, 160], "q_w": 1, "k_w_mk": 100},
    ]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="overlap"):
        cfg.load(bad)


def test_component_outside_domain_rejected(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["components"] = [
        {"name": "Out", "x_mm": [400, 410], "y_mm": [10, 20], "q_w": 1, "k_w_mk": 100},
    ]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="out of domain"):
        cfg.load(bad)
