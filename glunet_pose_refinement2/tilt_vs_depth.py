#!/usr/bin/env python3
"""
tilt_vs_depth.py - measure the tilt error of every cached board pose against
the table plane seen by the depth camera.

The board lies flat on the table, so its z-axis must be parallel to the table
normal. Both are measurable in the *camera* frame, so this needs no hand-eye
and no labels:

    n_vision = R_cam_board[:, 2]          (from the GLU-Net / PnP board pose)
    n_depth  = RANSAC plane normal        (from depth, sampled on the table
                                           OUTSIDE the board polygon)

    tilt_err = angle(n_vision, n_depth)

Depth is only used for the *normal*, never the offset: a depth scale error maps
a plane to a parallel plane, so it cannot corrupt the direction. That makes this
immune to the 0.95 depth-scale hack and to the transparent board (which is
masked out anyway).

Tilt is the DoF blocking 0.2 mm: at 465 mm standoff, 0.056 deg of tilt is
0.2 mm of corner error, but the pipeline is currently at 0.5-1.0 deg.

Also reports the plane offset, which gives the camera-table standoff and the
apparent board thickness, as an independent scale check.
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

K_4K = np.array([[2251.950927734375, 0, 1931.4296875],
                 [0, 2251.7568359375, 1091.6837158203125],
                 [0, 0, 1]], np.float64)
S = np.array([1280 / 3840, 720 / 2160])
DOWN = 6           # depth subsample stride; masks are built on this grid
DILATE = 260 // DOWN   # exclude board + its shadow/rim from the table fit
BAND = 900 // DOWN     # keep a ring of table around the board


def table_plane(depth, quad4k):
    """RANSAC plane through table pixels in an annulus around the board."""
    d = depth[::DOWN, ::DOWN].astype(np.float64) / 1000.0
    inner = np.zeros(d.shape, np.uint8)
    cv2.fillPoly(inner, [(quad4k / DOWN).astype(np.int32)], 255)
    outer = cv2.dilate(inner, np.ones((BAND, BAND), np.uint8))
    inner = cv2.dilate(inner, np.ones((DILATE, DILATE), np.uint8))
    ring = (outer > 0) & (inner == 0)

    m = ring & (d > 0.25) & (d < 1.2)
    ys, xs = np.nonzero(m)
    if len(xs) < 2000:
        return None, 0
    z = d[ys, xs]
    x = (xs * DOWN - K_4K[0, 2]) / K_4K[0, 0] * z
    y = (ys * DOWN - K_4K[1, 2]) / K_4K[1, 1] * z
    P = np.stack([x, y, z], 1)

    rng = np.random.default_rng(0)
    best_in, best = None, -1
    for _ in range(300):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        inl = np.abs((P - s[0]) @ n) < 0.004
        if inl.sum() > best:
            best, best_in = inl.sum(), inl
    if best_in is None or best < 1000:
        return None, 0
    Q = P[best_in]
    c = Q.mean(0)
    # smallest-eigenvector of the 3x3 scatter (never form the full SVD: with
    # ~60k points a full_matrices SVD would allocate a 60k x 60k U)
    n = np.linalg.eigh((Q - c).T @ (Q - c))[1][:, 0]
    if n[2] < 0:
        n = -n
    return (n, -n @ c), int(best)


def main():
    cache = json.load(open(Path(__file__).parent / "out" / "cached_views.json"))
    data = ROOT / "images" / "images_depth"

    lab_dir = Path(__file__).parent / "out_annotated"
    lab = {f.stem.split("_to_")[1]: c
           for c in ("bad", "close", "pixel-perfect")
           for f in (lab_dir / c).glob("match_*.png")}

    rows = []
    for p in sorted(cache):
        for view, rec in sorted(cache[p].items()):
            T = rec.get("T_cam_board")
            if T is None:
                continue
            vd = data / f"pose_{p}" / view
            depth = cv2.imread(str(vd / "depth.png"), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            T = np.asarray(T)
            rv, _ = cv2.Rodrigues(T[:3, :3])
            tv = T[:3, 3].reshape(3, 1)
            c = project(CORNER_3D, rv, tv) / S
            quad = np.array([c[0], c[1], c[3], c[2]])

            pl, n_inl = table_plane(depth, quad)
            if pl is None:
                continue
            n_d, d_d = pl
            n_v = T[:3, 2] * (1 if T[2, 2] > 0 else -1)
            if n_d @ n_v < 0:
                n_d, d_d = -n_d, -d_d

            tilt = np.degrees(np.arccos(np.clip(n_d @ n_v, -1, 1)))
            # signed height of the board origin above the depth table plane
            h_board = float(n_d @ T[:3, 3] + d_d)

            rows.append({
                "pose": int(p), "view": view, "label": lab.get(view, "-"),
                "tilt_deg": round(float(tilt), 3),
                "board_above_table_mm": round(h_board * 1000, 1),
                "cam_to_table_mm": round(float(-d_d) * 1000, 1),
                "cam_to_board_mm": round(float(T[2, 3]) * 1000, 1),
                "n_depth": [float(v) for v in n_d],
                "d_depth": float(d_d),
                "n_inliers": n_inl,
            })
            r = rows[-1]
            print(f"  pose_{p}/{view:<16} {r['label']:<14} tilt {tilt:6.3f} deg   "
                  f"board-table {r['board_above_table_mm']:+7.1f} mm   "
                  f"cam-table {r['cam_to_table_mm']:6.1f} mm")

    if not rows:
        print("nothing scored")
        return
    json.dump(rows, open(Path(__file__).parent / "out" / "tilt_vs_depth.json", "w"), indent=2)

    print("\n" + "=" * 70)
    t = np.array([r["tilt_deg"] for r in rows])
    print(f"tilt error over {len(rows)} views: median {np.median(t):.3f} deg  "
          f"mean {t.mean():.3f}  p90 {np.percentile(t, 90):.3f}  max {t.max():.3f}")
    print(f"  -> median corner error from tilt alone: "
          f"{np.tan(np.radians(np.median(t))) * 0.2032 * 1000:.2f} mm "
          f"(half-diagonal 203 mm)")
    for p in sorted({r["pose"] for r in rows}):
        s = [r for r in rows if r["pose"] == p]
        tt = np.array([r["tilt_deg"] for r in s])
        hh = np.array([r["board_above_table_mm"] for r in s])
        print(f"  pose_{p}: n={len(s):2d}  tilt med {np.median(tt):.3f} deg  "
              f"board-above-table {np.median(hh):+6.1f} +- {hh.std():.1f} mm")
    print("\nby label:")
    for c in ("pixel-perfect", "close", "bad", "-"):
        s = [r["tilt_deg"] for r in rows if r["label"] == c]
        if s:
            print(f"  {c:<14} n={len(s):2d}  median tilt {np.median(s):.3f} deg   "
                  f"range [{min(s):.3f}, {max(s):.3f}]")


if __name__ == "__main__":
    main()
