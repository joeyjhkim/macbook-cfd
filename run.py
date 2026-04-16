#!/usr/bin/env python3
"""CLI entry: load config -> build geometry -> NS solve -> energy solve -> figures + validate."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cfd import config as cfg
from cfd import energy, flow, geometry, validate, visualize


def main() -> int:
    parser = argparse.ArgumentParser(description="MacBook Pro 16\" 2D CFD")
    parser.add_argument("--config", "-c", type=Path,
                        default=Path("config/macbook_pro_16.yaml"),
                        help="Path to YAML config file")
    parser.add_argument("--output", "-o", type=Path, default=Path("figures"),
                        help="Output directory for figures")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation (faster, useful for tests)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress per-iteration solver logging")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    log = (lambda _msg: None) if args.quiet else print

    print(f"Loading config: {args.config}")
    config = cfg.load(args.config)
    print(f"  {len(config.components)} components, "
          f"grid {config.domain.nx}x{config.domain.ny}, "
          f"L={config.domain.length_m}m H={config.domain.height_m}m")

    print("Building geometry ...")
    geom = geometry.build(config)
    print(f"  Solid cells: {int(geom.is_solid.sum()):>6}  "
          f"Fan cells: {int(geom.is_fan.sum()):>4}  "
          f"Flow cells: {int(geom.flow_mask.sum()):>6}  "
          f"j_inlet={geom.j_inlet}")

    if not args.no_figures:
        out = args.output / "layout_validation.png"
        visualize.layout(config, geom, out)
        print(f"  layout figure -> {out}")

    print("Solving Navier–Stokes ...")
    t0 = time.time()
    flow_field = flow.solve(config, geom, log=log)
    print(f"  NS done in {time.time() - t0:.1f}s "
          f"({flow_field.iterations} iters, residual {flow_field.final_residual:.2e})")
    print(f"  Vmax = {flow_field.speed_cell.max():.2f} m/s")

    print("Solving energy ...")
    t0 = time.time()
    T = energy.solve(config, geom, flow_field)
    print(f"  Energy done in {time.time() - t0:.1f}s")
    print(f"  T range = [{T.T.min():.1f}, {T.T.max():.1f}] C")

    stats = validate.compute_stats(config, geom, T)
    Re = (config.fluid.rho_kg_m3 * config.bc.inlet_velocity_m_s
          * config.domain.length_m / config.fluid.mu_pa_s)
    print()
    print(f"  Reynolds = {Re:.0f}    Vmax = {flow_field.speed_cell.max():.2f} m/s")

    report = validate.validate(config, stats)
    for line in report.summary_lines():
        print(line)

    if not args.no_figures:
        visualize.temperature(config, geom, T,
                              args.output / "macbook_temperature.png", stats=stats)
        visualize.velocity(config, geom, flow_field,
                           args.output / "macbook_velocity.png")
        visualize.vorticity(config, geom, flow_field,
                            args.output / "macbook_vorticity.png")
        print("Figures written: layout / temperature / velocity / vorticity .png")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
