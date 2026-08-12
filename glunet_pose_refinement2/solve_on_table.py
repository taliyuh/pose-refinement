#!/usr/bin/env python3
"""
solve_on_table.py - re-solve every cached board pose with the out-of-plane
tilt LOCKED to the table plane measured by the depth camera.

Diagnosis this acts on: the pipeline solves the board pose from 4 coplanar
corners. Out-of-plane tilt is nearly unobservable in that configuration (a
near-fronto-parallel 320x250 board at ~460 mm), so sub-pixel corner noise turns
into tens of degrees of tilt. Measured against the depth table normal, the
cached poses are off by a median 1.3 deg and up to 51 deg -- and it is
uncorrelated with how good the mask looks, which is why mask quality has never
predicted pose error.

The board lies flat on the table, so tilt does not need to be estimated at all.
Fix it from depth (a 60k-point plane fit, far better conditioned than 4 corners)
and solve only what is well conditioned:

    R = R_plane @ Rz(yaw)
    t = t_plane + u*x_plane + v*y_plane + h*n

4 free parameters against 8 observations, so the fit can no longer hide corner
error by tilting -- the reprojection residual becomes a meaningful quality
signal instead of being identically zero.

Two things are then checkable with no hand-eye and no ground truth:
  * residual_px   - how well a rigid board on the table explains the corners
  * h             - the board's height above the table. It is a physical
                    constant, so its scatter across views measures the pose
                    accuracy in the depth direction directly.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "refinement_proposal"))
from pose_lib_improved import CORNER_3D, project  # noqa: E402

tilt = {(r["pose"], r["view"]): r
        for r in json.load(open(Path(__file__).parent / "out" / "tilt_vs_depth.json"))}
cache = json.load(open(Path(__file__).parent / "out" / "cached_views.json"))


def plane_frame(n, d):
    """Right-handed frame with z = n, origin = foot of the camera centre."""
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, n); x /= np.linalg.norm(x)
    y = np.cross(n, x)
    return np.stack([x, y, n], 1), -d * n


rows = []
for p in sorted(cache):
    for view, rec in sorted(cache[p].items()):
        T = rec.get("T_cam_board")
        key = (int(p), view)
        if T is None or key not in tilt:
            continue
        r = tilt[key]
        T = np.asarray(T)
        rv, _ = cv2.Rodrigues(T[:3, :3])
        px = project(CORNER_3D, rv, T[:3, 3].reshape(3, 1))   # corners GLU-Net gave

        n = np.array(r["n_depth"]); d = r["d_depth"]
        # orient the normal towards the camera
        if -d * n[2] > 0:
            n, d = -n, -d
        if n[2] > 0:
            n, d = -n, -d
        Rp, tp = plane_frame(n, d)

        def resid(q):
            yaw, u, v, h = q
            c, s = np.cos(yaw), np.sin(yaw)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            R = Rp @ Rz
            t = tp + Rp[:, 0] * u + Rp[:, 1] * v + n * h
            rvq, _ = cv2.Rodrigues(R)
            return (project(CORNER_3D, rvq, t.reshape(3, 1)) - px).ravel()

        best = None
        for yaw0 in np.linspace(-np.pi, np.pi, 8, endpoint=False):
            # initialise u,v by projecting the cached board origin into the plane frame
            o = Rp.T @ (T[:3, 3] - tp)
            s = least_squares(resid, [yaw0, o[0], o[1], o[2]], method="lm")
            if best is None or s.cost < best.cost:
                best = s
        f = best.fun.reshape(-1, 2)
        rows.append({
            "pose": int(p), "view": view, "label": r["label"],
            "tilt_before_deg": r["tilt_deg"],
            "resid_px": round(float(np.sqrt((best.fun ** 2).mean() * 2)), 3),
            "max_px": round(float(np.linalg.norm(f, axis=1).max()), 3),
            "h_mm": round(float(best.x[3]) * 1000, 2),
            "yaw_deg": round(float(np.degrees(best.x[0])) % 360, 2),
        })

print(f"{'view':<34}{'label':<15}{'tilt was':>9}{'resid px':>10}{'max px':>9}{'h mm':>9}")
print("-" * 88)
for r in sorted(rows, key=lambda x: -x["resid_px"]):
    print(f"pose_{r['pose']}/{r['view']:<26}{r['label']:<15}{r['tilt_before_deg']:9.2f}"
          f"{r['resid_px']:10.3f}{r['max_px']:9.3f}{r['h_mm']:9.1f}")

json.dump(rows, open(Path(__file__).parent / "out" / "on_table_fit.json", "w"), indent=2)
res = np.array([r["resid_px"] for r in rows])
print(f"\nreprojection residual of the 4-DoF on-table fit over {len(rows)} views:")
print(f"  median {np.median(res):.2f} px   p90 {np.percentile(res, 90):.2f}   max {res.max():.2f}")
print("\nboard height above the depth table plane (should be one constant):")
for p in sorted({r["pose"] for r in rows}):
    h = np.array([r["h_mm"] for r in rows if r["pose"] == p])
    g = np.array([r["h_mm"] for r in rows if r["pose"] == p and r["resid_px"] < 2])
    print(f"  pose_{p}: all n={len(h):2d} median {np.median(h):7.1f} sd {h.std():6.1f} mm"
          + (f"   |  resid<2px n={len(g):2d} median {np.median(g):7.1f} sd {g.std():5.2f} mm"
             if len(g) > 1 else ""))
print("\nresidual vs the mask labels:")
for c in ("pixel-perfect", "close", "bad"):
    s = [r["resid_px"] for r in rows if r["label"] == c]
    if s:
        print(f"  {c:<14} n={len(s):2d}  median {np.median(s):6.2f} px   "
              f"range [{min(s):.2f}, {max(s):.2f}]")
