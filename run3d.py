#!/usr/bin/env python3
"""3D CFD driver: load → geometry → 3D NS → 3D energy → figures + validate."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cfd3d import config3d as cfg_mod
from cfd3d import energy3d, flow3d, geometry3d, validate3d, visualize3d


def main() -> int:
    parser = argparse.ArgumentParser(description="MacBook Pro 16\" 3D CFD")
    parser.add_argument("--config", "-c", type=Path,
                        default=Path("config/macbook_pro_16_3d.yaml"))
    parser.add_argument("--output", "-o", type=Path, default=Path("figures"))
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    log = (lambda _m: None) if args.quiet else print

    print(f"Loading config: {args.config}")
    config = cfg_mod.load(args.config)
    d = config.domain
    print(f"  {len(config.components)} components, "
          f"grid {d.nx} x {d.ny} x {d.nz} "
          f"({d.nx*d.ny*d.nz} cells), "
          f"L={d.length_x_m}x{d.length_y_m}x{d.length_z_m}m")

    print("Building geometry ...")
    geom = geometry3d.build(config)
    print(f"  Solid cells: {int(geom.is_solid.sum()):>6}  "
          f"Fan cells: {int(geom.is_fan.sum()):>4}  "
          f"Flow cells: {int(geom.flow_mask.sum()):>6}")

    if not args.no_figures:
        out = args.output / "layout3d.png"
        visualize3d.layout(config, geom, out)
        print(f"  layout figure -> {out}")

    print("Solving 3D Navier–Stokes ...")
    t0 = time.time()
    flow = flow3d.solve(config, geom, log=log)
    print(f"  NS done in {time.time() - t0:.1f}s  "
          f"({flow.iterations} iters, div={flow.final_div:.2e}, "
          f"dU={flow.final_mom_res:.2e})")
    print(f"  Vmax = {flow.speed_cell.max():.2f} m/s")

    print("Solving 3D energy ...")
    t0 = time.time()
    T = energy3d.solve(config, geom, flow)
    print(f"  Energy done in {time.time() - t0:.1f}s")
    print(f"  T range = [{T.T.min():.1f}, {T.T.max():.1f}] C")

    stats = validate3d.compute_stats(config, geom, T)
    report = validate3d.validate(config, stats)
    print()
    for line in report.summary_lines():
        print(line)

    if not args.no_figures:
        visualize3d.temperature(config, geom, T,
                                args.output / "macbook_temperature3d.png", stats=stats)
        visualize3d.velocity(config, geom, flow,
                             args.output / "macbook_velocity3d.png")
        print("Figures written: layout3d / temperature3d / velocity3d .png")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
