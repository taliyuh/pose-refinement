#!/usr/bin/env python3
"""
eval_multiview.py - does the multi-view fit actually beat pose averaging?

Runs the split-half test of fit_board_multiview.py over every folder: fit two
disjoint halves of the views independently, compare the two board poses. The
board never moved, so the disagreement IS the error, doubled -- and nothing
external is involved, so nothing can flatter either estimator.

Both estimators see exactly the same views, so the comparison is between models
and not between datasets.

    --sweep   also scan --chain-sigma, the one knob. It says how far the robot
              chain is trusted; too small collapses to the rigid fit (bad, the
              chain is 7x noisier than the vision), too large throws away the
              cross-view constraint. 0.40 mm won this sweep and is the
              default in fit_board_multiview.py; re-run this after any change
              to the bundle or the kinematic calibration.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import board_model  # noqa: E402
from fit_board_multiview import (average_per_view, halves, load,  # noqa: E402
                                 pose_delta, solve)

POSES = [4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16]


def split_half(views, intr, X0, geom, chain_sigma, bias=None, refine_x=False):
    """(joint-fit disagreement, averaging disagreement) in mm at the corner."""
    A, B = halves(views)
    if len(A) < 2 or len(B) < 2:
        return None
    pts3, nrm2, level, loop = geom
    Ta, _, _, _ = solve(A, intr, X0, pts3, nrm2, level, loop, bias,
                        chain_sigma=chain_sigma, refine_x=refine_x, verbose=False)
    Tb, _, _, _ = solve(B, intr, X0, pts3, nrm2, level, loop, bias,
                        chain_sigma=chain_sigma, refine_x=refine_x, verbose=False)
    Ma, _ = average_per_view(A, X0)
    Mb, _ = average_per_view(B, X0)
    return pose_delta(Ta, Tb), pose_delta(Ma, Mb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", default=",".join(map(str, POSES)))
    ap.add_argument("--bundle", default=str(HERE / "out" / "bundle_rims6_v3.json"))
    ap.add_argument("--calib", default=str(HERE / "out" / "kinematic_calib_v3.json"))
    ap.add_argument("--model", default="joint offsets only")
    ap.add_argument("--max-radius", type=float, default=500.0)
    ap.add_argument("--sweep", action="store_true",
                    help="scan --chain-sigma instead of using the default")
    ap.add_argument("--sigmas", default="0.4,0.87,2.0,5.0",
                    help="chain_sigma values [mm] for --sweep")
    ap.add_argument("--out", default=str(HERE / "out" / "eval_multiview.json"))
    args = ap.parse_args()

    poses = [int(s) for s in args.poses.split(",")]
    cal = json.load(open(args.calib))[args.model]
    dtheta = np.radians(cal["dtheta_deg"])
    X0 = np.asarray(cal["X_tcp_cam"])
    from fit_board_multiview import DEFAULT_CHAIN_SIGMA
    default_sigma = DEFAULT_CHAIN_SIGMA
    geom = board_model.build_levels()
    data = ROOT / "images" / "images_depth"

    print(f"kinematic model '{args.model}', chain_sigma {default_sigma:.2f} mm "
          f"(kinematic_calib residual {cal.get('median_mm', float('nan')):.3f} mm)")
    loaded = {}
    for p in poses:
        v, intr, bias = load(p, args.bundle, dtheta, data, args.max_radius, quiet=True)
        if len(v) >= 4:
            loaded[p] = (v, intr, bias)
        else:
            print(f"  pose_{p}: only {len(v)} usable views, skipped")
    print(f"{len(loaded)} folders, "
          f"{sum(len(v) for v, _, _ in loaded.values())} views\n")

    sigmas = ([float(t) for t in args.sigmas.split(",")] if args.sweep
              else [default_sigma])
    table = {}
    for sig in sigmas:
        rows = []
        print(f"--- chain_sigma = {sig:.2f} mm ---")
        print(f"  {'folder':10} {'n':>3}   {'joint fit':>10}  {'averaging':>10}   ratio")
        for p, (views, intr, bias) in loaded.items():
            r = split_half(views, intr, X0, geom, sig, bias)
            if r is None:
                continue
            (dj, tj, rj), (da, ta, ra) = r
            rows.append({"pose": p, "n": len(views), "joint_mm": dj,
                         "avg_mm": da, "joint_rot_deg": rj, "avg_rot_deg": ra})
            print(f"  pose_{p:<5} {len(views):3d}   {dj:10.3f}  {da:10.3f}   "
                  f"{da / dj:5.2f}x" + ("" if dj < da else "   <- worse"))
        j = np.array([r["joint_mm"] for r in rows])
        a = np.array([r["avg_mm"] for r in rows])
        print(f"  {'median':10} {'':3}   {np.median(j):10.3f}  {np.median(a):10.3f}"
              f"   {np.median(a) / np.median(j):5.2f}x     "
              f"({(j < a).sum()}/{len(j)} folders improved)\n")
        table[f"{sig:.2f}"] = {"rows": rows,
                               "median_joint_mm": float(np.median(j)),
                               "median_avg_mm": float(np.median(a)),
                               "folders_improved": int((j < a).sum()),
                               "folders": len(j)}

    if args.sweep:
        best = min(table, key=lambda k: table[k]["median_joint_mm"])
        print(f"best chain_sigma: {best} mm at "
              f"{table[best]['median_joint_mm']:.3f} mm "
              f"(default {default_sigma:.2f} gives "
              f"{table[f'{default_sigma:.2f}']['median_joint_mm']:.3f} mm)"
              if f"{default_sigma:.2f}" in table else
              f"best chain_sigma: {best} mm")

    json.dump(table, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
