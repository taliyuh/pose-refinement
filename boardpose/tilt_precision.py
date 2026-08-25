#!/usr/bin/env python3
"""
tilt_precision.py - does the second contour level actually fix tilt?

Tilt was the entire remaining error budget: two independent estimators (tilt
from depth vs tilt from a single-plane image fit) disagreed by 0.57 deg, which
is 2.0 mm of out-of-plane corner displacement, against a 0.056 deg budget.

Measured here with no external reference: fit each view twice from disjoint,
spatially interleaved halves of the pockets, and compare. Done for the
single-plane model (top rims only) and the two-level model (top rims + pocket
floors 22 mm below), so the difference isolates the effect of the baseline.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

import board_model  # noqa: E402
from bundle_rims import load_views, project_with_normals  # noqa: E402
from camera import edge_search  # noqa: E402

HALF_DIAG = 0.2032


def interleave(pts3, loop, level):
    """Alternating spatial parity, consistent between levels."""
    par = np.zeros(len(pts3), int)
    for li in np.unique(level):
        m = level == li
        ls = np.unique(loop[m])
        cx = np.array([pts3[m & (loop == g)][:, 0].mean() for g in ls])
        rank = np.argsort(np.argsort(cx))
        for g, r in zip(ls, rank):
            par[m & (loop == g)] = r % 2
    return par


def fit(v, intr, pts3, nrm2, level, loop, x0, schedule=(6, 4, 4)):
    x = np.array(x0, float)
    for S in schedule:
        pm, d = project_with_normals(pts3, nrm2, x[:3], x[3:], intr)
        vis = board_model.visible(pts3, nrm2, level, loop,
                                  cv2.Rodrigues(x[:3])[0], x[3:])
        loc = pm - [v.ox, v.oy]
        off, peak, ok = edge_search(v.gx, v.gy, loc, d, S)
        h, w = v.gx.shape
        ok &= vis & (loc[:, 0] > S + 2) & (loc[:, 0] < w - S - 2) \
            & (loc[:, 1] > S + 2) & (loc[:, 1] < h - S - 2)
        if ok.sum() < 80:
            return None
        ok &= peak >= np.percentile(peak[ok], 25)
        q, dd = (pm + off[:, None] * d)[ok], d[ok]
        idx = np.nonzero(ok)[0]

        def res(z):
            a, _ = project_with_normals(pts3[idx], nrm2[idx], z[:3], z[3:], intr)
            return ((a - q) * dd).sum(1)

        sol = least_squares(res, x, loss="soft_l1", f_scale=1.5,
                            x_scale=[0.01, 0.01, 0.01, 0.002, 0.002, 0.002],
                            max_nfev=50)
        x = sol.x
    return x


def tilt_yaw_diff(xa, xb):
    Ra, Rb = cv2.Rodrigues(xa[:3])[0], cv2.Rodrigues(xb[:3])[0]
    dR = Ra.T @ Rb
    ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
    rv = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
    rv = rv / max(np.linalg.norm(rv), 1e-12) * ang
    dt = Ra.T @ (xb[3:] - xa[3:]) * 1000
    return (np.degrees(np.hypot(rv[0], rv[1])), abs(np.degrees(rv[2])),
            float(np.hypot(dt[0], dt[1])), abs(float(dt[2])))


def main():
    B = json.load(open(HERE / "out" / "bundle_2level.json"))
    intr = np.array(B["intrinsics"])
    init = {(r["pose"], r["view"]): np.asarray(r["T_cam_board"]) for r in B["views"]}
    views = load_views(HERE / "out" / "dense_on_table.json", 1.5,
                       ROOT / "images" / "images_depth")

    p3, n3, lv3, lp3 = board_model.build_levels()
    p1, n1, lp1 = board_model.build()
    p1 = board_model.pts3(p1); lv1 = np.zeros(len(p1), int)
    models = {"top rims only (1 plane)": (p1, n1, lv1, lp1, interleave(p1, lp1, lv1)),
              "top + floor rims (22mm)": (p3, n3, lv3, lp3, interleave(p3, lp3, lv3))}

    res = {k: [] for k in models}
    for v in views:
        T = init.get((v.pose, v.name))
        if T is None:
            continue
        x0 = np.concatenate([cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]])
        for name, (P, N, L, G, par) in models.items():
            out = []
            for pa in (0, 1):
                m = par == pa
                out.append(fit(v, intr, P[m], N[m], L[m], G[m], x0))
            if any(o is None for o in out):
                continue
            res[name].append(tilt_yaw_diff(*out))

    print("half-vs-half disagreement, 6-DoF free (no depth, no marker, no GT)\n")
    hdr = (f"{'model':<26}{'n':>4}{'tilt deg':>10}{'-> mm':>8}"
           f"{'yaw deg':>10}{'-> mm':>8}{'in-plane mm':>13}{'normal mm':>11}")
    print(hdr); print("-" * len(hdr))
    for name, r in res.items():
        if not r:
            continue
        a = np.abs(np.array(r))
        med = np.median(a, axis=0)
        print(f"{name:<26}{len(a):>4}{med[0]:>10.4f}"
              f"{np.radians(med[0]) * HALF_DIAG * 1000:>8.3f}"
              f"{med[1]:>10.4f}{np.radians(med[1]) * HALF_DIAG * 1000:>8.3f}"
              f"{med[2]:>13.3f}{med[3]:>11.3f}")
    print("\n(each half uses half the pockets, so a full fit is ~sqrt(2) better)")
    json.dump({k: np.asarray(v).tolist() for k, v in res.items()},
              open(HERE / "out" / "tilt_precision.json", "w"), indent=2)


if __name__ == "__main__":
    main()
