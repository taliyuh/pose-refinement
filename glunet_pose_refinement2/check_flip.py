#!/usr/bin/env python3
"""
check_flip.py - test whether the large tilt errors are the planar-PnP
two-solution ambiguity, and whether the depth table normal picks the right one.

Coplanar PnP has two solutions that reproject identically. The pipeline solves
the board pose from 4 coplanar corners (solve_corners -> SOLVEPNP_ITERATIVE),
so it can silently return the mirrored one. The mask still looks pixel-perfect,
because both solutions put the corners in the same place -- which is exactly why
mask quality does not predict pose error, and why a "close" view can beat a
"pixel-perfect" one.

Method: reproject the cached pose's corners, re-solve with IPPE to recover BOTH
solutions, and score each against the table normal measured by the depth camera.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "refinement_proposal"))
from pose_lib_improved import CORNER_3D, project  # noqa: E402
from calib import K, DIST  # noqa: E402

Z = float(CORNER_3D[0, 2])
OBJ0 = np.ascontiguousarray(CORNER_3D - [0, 0, Z], np.float32)

tilt = {(r["pose"], r["view"]): r
        for r in json.load(open(Path(__file__).parent / "out" / "tilt_vs_depth.json"))}
cache = json.load(open(Path(__file__).parent / "out" / "cached_views.json"))


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1))))


rows = []
for p in sorted(cache):
    for view, rec in sorted(cache[p].items()):
        T = rec.get("T_cam_board")
        key = (int(p), view)
        if T is None or key not in tilt:
            continue
        T = np.asarray(T)
        rv, _ = cv2.Rodrigues(T[:3, :3])
        px = project(CORNER_3D, rv, T[:3, 3].reshape(3, 1))

        n_sol, rvs, tvs, _ = cv2.solvePnPGeneric(
            OBJ0, np.ascontiguousarray(px, np.float32),
            np.asarray(K, np.float64), np.asarray(DIST, np.float64),
            flags=cv2.SOLVEPNP_IPPE)
        if n_sol < 2:
            continue

        r = tilt[key]
        n_d = np.array(r["n_depth"])
        sols = []
        for i in range(n_sol):
            R, _ = cv2.Rodrigues(rvs[i])
            t_i = tvs[i].ravel() - Z * R[:, 2]
            rp = project(CORNER_3D, rvs[i], t_i.reshape(3, 1))
            sols.append({"tilt": ang(R[:, 2], n_d),
                         "reproj": float(np.abs(rp - px).max()),
                         "tz": float(t_i[2]) * 1000})
        best = int(np.argmin([s["tilt"] for s in sols]))
        rows.append({
            "pose": int(p), "view": view, "label": r["label"],
            "tilt_cached": r["tilt_deg"],
            "tilt_sols": [round(s["tilt"], 2) for s in sols],
            "reproj": [round(s["reproj"], 3) for s in sols],
            "tz": [round(s["tz"], 1) for s in sols],
            "picked_wrong": r["tilt_deg"] > min(s["tilt"] for s in sols) + 2.0,
            "best_tilt": round(min(s["tilt"] for s in sols), 3),
            "dz_between_sols_mm": round(abs(sols[0]["tz"] - sols[1]["tz"]), 1),
        })

print(f"{'view':<34}{'label':<15}{'cached':>7}{'sol tilts':>16}"
      f"{'reproj px':>18}{'  flip?'}")
print("-" * 100)
for r in sorted(rows, key=lambda x: -x["tilt_cached"]):
    print(f"pose_{r['pose']}/{r['view']:<26}{r['label']:<15}{r['tilt_cached']:7.2f}"
          f"{str(r['tilt_sols']):>16}{str(r['reproj']):>18}"
          f"   {'FLIPPED' if r['picked_wrong'] else ''}")

n_flip = sum(r["picked_wrong"] for r in rows)
print(f"\n{n_flip}/{len(rows)} views took the WRONG branch of the planar ambiguity")
mx = max(max(r["reproj"]) for r in rows)
print(f"max reprojection error of either solution: {mx:.3f} px "
      f"-> the two solutions are indistinguishable in the image")
bt = np.array([r["best_tilt"] for r in rows])
print(f"\ntilt after picking the depth-consistent branch: median {np.median(bt):.3f} deg "
      f"p90 {np.percentile(bt, 90):.3f}  max {bt.max():.3f}")
print(f"  corner error from median tilt: "
      f"{np.tan(np.radians(np.median(bt))) * 0.2032 * 1000:.2f} mm  (was "
      f"{np.tan(np.radians(np.median([r['tilt_cached'] for r in rows]))) * 0.2032 * 1000:.2f} mm)")
print(f"\ndepth between the two branches: median {np.median([r['dz_between_sols_mm'] for r in rows]):.1f} mm"
      f"  max {max(r['dz_between_sols_mm'] for r in rows):.1f} mm")
for c in ("pixel-perfect", "close", "bad"):
    s = [r for r in rows if r["label"] == c]
    if s:
        print(f"  {c:<14} n={len(s):2d}  flipped {sum(x['picked_wrong'] for x in s):2d}")
json.dump(rows, open(Path(__file__).parent / "out" / "flip_check.json", "w"), indent=2)
