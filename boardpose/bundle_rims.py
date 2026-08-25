#!/usr/bin/env python3
"""
bundle_rims.py - multi-view bundle adjustment over the hole-rim observations.

What is actually shared between views: the board is static within a folder, but
its geometry is already known exactly from the STL, so each view's pose is
independently determined. The genuinely shared unknown is the CAMERA ITSELF.

So this solves, jointly over every view of every folder:

    shared      fx, fy, cx, cy, k1, k2, p1, p2, k3        (9)
    shared      rim bias, per contour level                (2-6)
    per view    full 6-DoF T_cam_board                    (6 x N)

against ~50k subpixel rim-edge observations. That is Zhang-style calibration
carried out on the actual measurement object rather than a separate
checkerboard, which is what removes the systematic bias that no amount of
per-view fitting can reach. The current K_4K is exactly 3x the 720p matrix,
i.e. it was rescaled rather than recalibrated -- this checks it.

Two other things change versus the old on-table fitter:

  * tilt is FREE here. The table constraint was needed because 4 corners cannot
    see tilt; 1184 subpixel points over a 320 x 250 mm board at 37 deg angular
    extent can. Depth is then demoted from an input to an independent check.
  * distortion is estimated, so points are projected INTO the raw image and the
    edge search runs on unresampled pixels.
  * the RIM BIAS is estimated. The STL outline is not where the intensity edge
    is -- the top hole edge is chamfered and the pocket floor rim sits at the
    base of a 22 mm wall -- and because tilt is triangulated from the 22 mm
    baseline between the two levels, the differential bias between them is a
    tilt error by construction. It was worth 0.68 deg of tilt bias and 0.18 deg
    of tilt scatter, the latter being the entire per-view tilt error seen in the
    robot frame. See board_model.rim_bias(). It is identifiable here rather than
    degenerate with focal length because a radial offset changes each rim's SIZE
    without changing the SPACING between rims, which a scale change cannot do.

The marker is deliberately kept out of the bundle: folding in T_cam_marker
would contaminate the board poses with ArUco error. It is used afterwards, as
an independent check, exactly as before.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import board_model  # noqa: E402
from camera import K_4K, D_4K, edge_search  # noqa: E402

NINTR = 9
NBIAS = board_model.NBIAS
MARGIN = 140


# ------------------------------------------------------------------ project

def project(P3, rvec, tvec, intr):
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = intr
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    X = P3 @ R.T + tvec
    z = np.maximum(X[:, 2], 1e-6)
    x, y = X[:, 0] / z, X[:, 1] / z
    r2 = x * x + y * y
    rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([fx * xd + cx, fy * yd + cy], 1)


def project_with_normals(pts3, nrm2, rvec, tvec, intr):
    off = np.concatenate([nrm2 * 1e-3, np.zeros((len(nrm2), 1))], 1)
    a = project(pts3, rvec, tvec, intr)
    b = project(pts3 + off, rvec, tvec, intr)
    d = b - a
    return a, d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)


# ------------------------------------------------------------------ views

class View:
    __slots__ = ("pose", "name", "label", "gx", "gy", "ox", "oy", "rvec", "tvec")


def load_views(init_json, max_mad, data):
    rows = [r for r in json.load(open(init_json)) if r["mad_px"] < max_mad]
    out = []
    for r in rows:
        d = data / f"pose_{r['pose']}" / r["view"]
        img = cv2.imread(str(d / "color.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        T = np.asarray(r["T_cam_board"])
        v = View()
        v.pose, v.name, v.label = r["pose"], r["view"], r["label"]
        v.rvec = cv2.Rodrigues(T[:3, :3])[0].ravel()
        v.tvec = T[:3, 3].copy()

        # crop to the board plus margin so 67 gradient images fit in RAM
        c = project(board_model.build_levels()[0], v.rvec, v.tvec,
                    list(np.diag(K_4K)[:2]) + [K_4K[0, 2], K_4K[1, 2]] + list(D_4K))
        x0 = int(max(0, c[:, 0].min() - MARGIN)); x1 = int(min(img.shape[1], c[:, 0].max() + MARGIN))
        y0 = int(max(0, c[:, 1].min() - MARGIN)); y1 = int(min(img.shape[0], c[:, 1].max() + MARGIN))
        if x1 - x0 < 200 or y1 - y0 < 200:
            continue
        g = cv2.GaussianBlur(img[y0:y1, x0:x1], (0, 0), 1.6)
        v.gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
        v.gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
        v.ox, v.oy = x0, y0
        out.append(v)
    return out


# ------------------------------------------------------------------ bundle

def measure(views, intr, pts3, nrm2, S, level=None, loop=None, bias=None):
    """One ICP-style association pass: detect the edge each rim point sits on."""
    obs = []
    for v in views:
        R = cv2.Rodrigues(v.rvec)[0]
        P = (pts3 if bias is None or level is None else
             board_model.biased_pts(pts3, nrm2, level, R, v.tvec, bias))
        pm, d = project_with_normals(P, nrm2, v.rvec, v.tvec, intr)
        vis = (np.ones(len(pts3), bool) if level is None else
               board_model.visible(pts3, nrm2, level, loop, R, v.tvec))
        loc = pm - [v.ox, v.oy]
        off, peak, ok = edge_search(v.gx, v.gy, loc, d, S)
        h, w = v.gx.shape
        ok &= vis
        ok &= (loc[:, 0] > S + 2) & (loc[:, 0] < w - S - 2) \
            & (loc[:, 1] > S + 2) & (loc[:, 1] < h - S - 2)
        if ok.sum() > 50:
            ok &= peak >= np.percentile(peak[ok], 25)
        obs.append({"idx": np.nonzero(ok)[0],
                    "q": (pm + off[:, None] * d)[ok], "d": d[ok]})
    return obs


NSH = NINTR + NBIAS          # shared parameters, laid out first


def pack(views, intr, bias):
    return np.concatenate([intr, bias]
                          + [np.concatenate([v.rvec, v.tvec]) for v in views])


def unpack(x, views):
    for i, v in enumerate(views):
        s = NSH + 6 * i
        v.rvec, v.tvec = x[s:s + 3], x[s + 3:s + 6]
    return x[:NINTR], x[NINTR:NSH]


def residuals(x, views, obs, pts3, nrm2, level, fit_intr, intr0, free_bias, bias0):
    intr = x[:NINTR] if fit_intr else intr0
    # only the free bias components come from x; the rest stay at their seed,
    # so a zero-Jacobian column can never drift
    bias = np.asarray(bias0, float).copy()
    for b in free_bias:
        bias[b] = x[NINTR + b]
    out = []
    for i, (v, o) in enumerate(zip(views, obs)):
        s = NSH + 6 * i
        if len(o["idx"]) == 0:
            continue
        idx = o["idx"]
        rv, tv = x[s:s + 3], x[s + 3:s + 6]
        P = pts3[idx]
        if level is not None:
            P = board_model.biased_pts(P, nrm2[idx], level[idx],
                                       cv2.Rodrigues(rv)[0], tv, bias)
        a, _ = project_with_normals(P, nrm2[idx], rv, tv, intr)
        out.append(((a - o["q"]) * o["d"]).sum(1))
    return np.concatenate(out) if out else np.zeros(1)


def sparsity(views, obs, fit_intr, free_bias):
    n = sum(len(o["idx"]) for o in obs)
    J = lil_matrix((n, NSH + 6 * len(views)), dtype=np.uint8)
    r = 0
    for i, o in enumerate(obs):
        k = len(o["idx"])
        if k == 0:
            continue
        if fit_intr:
            J[r:r + k, :NINTR] = 1
        for b in free_bias:
            J[r:r + k, NINTR + b] = 1
        J[r:r + k, NSH + 6 * i: NSH + 6 * i + 6] = 1
        r += k
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(HERE / "out" / "dense_on_table.json"))
    ap.add_argument("--max-mad", type=float, default=1.5,
                    help="only seed from views the on-table fit already explains")
    ap.add_argument("--fix-intrinsics", action="store_true")
    ap.add_argument("--rim-bias", default="b0",
                    choices=["none", "b0", "b0sym", "b0b1"],
                    help="none: the pre-2026-08 model. "
                         "b0: constant offset per contour level (default). "
                         "b0sym: b0 plus the SYMMETRIC view terms b2|s| and "
                         "b3|u_xy| -- safe but USELESS, they improve the pixel "
                         "fit and make the pose worse. "
                         "b0b1: b0 plus the signed term -- REJECTED, b1*s "
                         "translates the rim. See board_model.rim_bias().")
    ap.add_argument("--top-only", action="store_true",
                    help="use only the top rims (single plane) for comparison")
    ap.add_argument("--out", default=str(HERE / "out" / "bundle_rims.json"))
    args = ap.parse_args()

    if args.top_only:
        p2, nrm2, loop = board_model.build()
        pts3 = board_model.pts3(p2); level = np.zeros(len(p2), int)
        # One level means no differential, so the rim bias has nothing to
        # triangulate against and b0 collapses into the standoff. This mode
        # exists only as the 1-plane comparison; keep it as it was.
        if args.rim_bias != "none":
            print("--top-only: rim bias is not identifiable from one level, "
                  "disabling it")
            args.rim_bias = "none"
    else:
        pts3, nrm2, level, loop = board_model.build_levels()
    data = ROOT / "images" / "images_depth"
    t0 = time.time()
    views = load_views(args.init, args.max_mad, data)
    print(f"{len(views)} views, {len(pts3)} rim points each "
          f"({time.time() - t0:.0f}s to load)")

    intr0 = np.array([K_4K[0, 0], K_4K[1, 1], K_4K[0, 2], K_4K[1, 2],
                      *D_4K.astype(np.float64)])
    intr = intr0.copy()
    fit_intr = not args.fix_intrinsics
    bias0 = board_model.RIM_BIAS0.copy()
    if args.rim_bias == "none":
        bias0 = np.zeros(NBIAS)
    bias = bias0.copy()
    free = {"none": [], "b0": [0, 1], "b0sym": [0, 1, 4, 5, 6, 7],
            "b0b1": [0, 1, 2, 3]}[args.rim_bias]

    for it, S in enumerate((10, 6, 4, 4)):
        obs = measure(views, intr, pts3, nrm2, S, level, loop, bias)
        x0 = pack(views, intr, bias)
        sp = sparsity(views, obs, fit_intr, free)
        sol = least_squares(residuals, x0, jac_sparsity=sp, loss="soft_l1",
                            f_scale=1.5, method="trf", max_nfev=40, verbose=0,
                            args=(views, obs, pts3, nrm2, level, fit_intr,
                                  intr0, free, bias0))
        fi, fb = unpack(sol.x, views)
        intr = fi if fit_intr else intr0
        bias = bias0.copy()
        for b in free:
            bias[b] = fb[b]
        r = sol.fun
        print(f"  iter {it} (S={S:2d})  n={len(r):6d}  "
              f"rms {np.sqrt((r ** 2).mean()):.3f} px  "
              f"mad {np.median(np.abs(r)):.3f} px  "
              f"in1px {(np.abs(r) < 1).mean():.2f}   [{time.time() - t0:.0f}s]")

    print("\nintrinsics" + ("  (refined)" if fit_intr else "  (held fixed)"))
    names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
    for nm, a, b in zip(names, intr0, intr):
        print(f"  {nm:<3} {a:>14.6f} -> {b:>14.6f}   delta {b - a:+.6f}")

    print(f"\nrim bias  (--rim-bias {args.rim_bias})"
          + "   [mm along each rim's outward normal]")
    for i, (nm, a, b) in enumerate(zip(board_model.RIM_BIAS_NAMES, bias0, bias)):
        tag = "" if i in free else "   (held)"
        print(f"  {nm:<9} {a * 1000:>9.4f} -> {b * 1000:>9.4f} mm{tag}")
    dif = (bias[1] - bias[0]) * 1000
    print(f"  differential b0 (floor - top): {dif:+.4f} mm across the 22 mm "
          f"level baseline = {np.degrees(np.arctan2(abs(dif), 22.0)):.4f} deg "
          f"of tilt it was costing")

    rows = []
    obs = measure(views, intr, pts3, nrm2, 4, level, loop, bias)
    for v, o in zip(views, obs):
        if len(o["idx"]) == 0:
            continue
        P = board_model.biased_pts(pts3[o["idx"]], nrm2[o["idx"]], level[o["idx"]],
                                   cv2.Rodrigues(v.rvec)[0], v.tvec, bias)
        a, _ = project_with_normals(P, nrm2[o["idx"]], v.rvec, v.tvec, intr)
        r = ((a - o["q"]) * o["d"]).sum(1)
        T = np.eye(4); T[:3, :3] = cv2.Rodrigues(v.rvec)[0]; T[:3, 3] = v.tvec
        rows.append({"pose": v.pose, "view": v.name, "label": v.label,
                     "n": int(len(r)), "rms_px": float(np.sqrt((r ** 2).mean())),
                     "mad_px": float(np.median(np.abs(r))),
                     "T_cam_board": T.tolist()})
    json.dump({"intrinsics": list(map(float, intr)),
               "intrinsics_names": names,
               "rim_bias": list(map(float, bias)),
               "rim_bias_names": board_model.RIM_BIAS_NAMES,
               "views": rows},
              open(args.out, "w"), indent=2)
    m = np.array([r["mad_px"] for r in rows])
    print(f"\n{len(rows)} views   rim residual MAD: median {np.median(m):.3f} px "
          f"({np.median(m) * 0.209:.3f} mm)   p90 {np.percentile(m, 90):.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
