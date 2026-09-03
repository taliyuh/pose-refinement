#!/usr/bin/env python3
"""
fit_rims6.py - full 6-DoF hole-rim fit, seeded per image by seed_board.py.

Replaces the old on-table fitter (`fit_dense_on_table.py`, removed: it solved
only yaw/u/v and needed an ArUco marker to seed) for captures with no marker in
frame. Two changes, both of which remove a dependency rather than add a model:

  1. No marker, no cached seed. seed_board.py produces ranked candidate poses
     from the image and its depth frame alone. Each candidate is refined here
     and the winner is chosen on the rim residual -- the quantity that is
     actually being minimised -- instead of on the chamfer score that ranked
     them.

  2. Tilt is FREE. The old fitter solved only yaw, u, v, and took roll,
     pitch and height from a depth plane whose ROI came from the seed pose. A
     wrong seed therefore produced a wrong plane, and no in-plane search could
     recover: measured on pose_12/131731_390724, widening the in-plane search
     from +-30 mm to +-150 mm with full 360 deg yaw moved the residual only
     1.72 -> 1.52 px, while freeing all 6 DoF reached 0.46 px on pose_16.

     Tilt is observable without depth here because the model is two-level: the
     pockets are blind, 22 mm deep, so the floor rims at z = 3 mm and the top
     rims at z = 25 mm triangulate against a real baseline. Depth is demoted to
     seeding and gating, which is where the README already places it (its table
     normal is 0.66 deg off the image-derived one).

Output feeds bundle_rims.py unchanged:  --init out/rims6.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import board_model  # noqa: E402
import seed_board  # noqa: E402
from camera import K_4K, D_4K, edge_search  # noqa: E402


# ---------------------------------------------------------------- projection

def project(P3, R, t):
    X = P3 @ R.T + t
    z = np.maximum(X[:, 2], 1e-6)
    return np.stack([K_4K[0, 0] * X[:, 0] / z + K_4K[0, 2],
                     K_4K[1, 1] * X[:, 1] / z + K_4K[1, 2]], 1)


def RT(x):
    R, _ = cv2.Rodrigues(np.asarray(x[:3], float).reshape(3, 1))
    return R, np.asarray(x[3:6], float)


def rim_points(P3, N2, R, t):
    """Projected rim points and the image direction of their outward normals.

    The normal is in-plane (the rims are planar curves at fixed z), so the 1 mm
    probe point is offset in x, y only.
    """
    Q = P3.copy()
    Q[:, :2] += N2 * 1e-3
    a, b = project(P3, R, t), project(Q, R, t)
    d = b - a
    return a, d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)


# ---------------------------------------------------------------- refinement

def refine(gx, gy, P3, N2, lvl, loop, x0, schedule=(24, 12, 6, 4, 4),
           two_level=True):
    """Iterated subpixel edge search + 6-DoF Gauss-Newton, shrinking the radius.

    Floor-rim points hidden behind the pocket wall are culled every round, since
    which of them are visible depends on the pose being estimated.
    """
    pose = np.array(x0, float)
    info = None
    for search_halfwidth in schedule:
        edges = _associate_edges(gx, gy, P3, N2, lvl, loop, pose,
                                 search_halfwidth, two_level)
        if edges is None:
            return pose, None

        point_idx, edge_xy, edge_normal = edges

        def res(candidate):
            R_candidate, t_candidate = RT(candidate)
            projected, _ = rim_points(P3[point_idx], N2[point_idx],
                                      R_candidate, t_candidate)
            return ((projected - edge_xy) * edge_normal).sum(1)

        solution = least_squares(res, pose, loss="soft_l1", f_scale=2.0,
                                 x_scale=[0.01, 0.01, 0.01,
                                          0.002, 0.002, 0.002],
                                 max_nfev=80)
        pose = solution.x
        info = _fit_quality(len(point_idx), solution.fun)
    return pose, info


MIN_RIM_POINTS = 100


def _associate_edges(gx, gy, P3, N2, lvl, loop, pose, search_halfwidth,
                     two_level):
    """(point indices, edge points, edge normals), or None if too few survive.

    Floor-rim points hidden behind the pocket wall are culled every round,
    since which of them are visible depends on the pose being estimated.
    """
    R, t = RT(pose)
    if two_level:
        is_visible = board_model.visible(P3, N2, lvl, loop, R, t)
    else:
        is_visible = np.ones(len(P3), bool)
    if is_visible.sum() < MIN_RIM_POINTS:
        return None

    visible_idx = np.nonzero(is_visible)[0]
    projected, normal = rim_points(P3[visible_idx], N2[visible_idx], R, t)
    offset, peak, is_usable = edge_search(gx, gy, projected, normal,
                                          search_halfwidth)
    if is_usable.sum() < MIN_RIM_POINTS:
        return None

    # Drop the weakest quarter: a rim point on a faint gradient is a worse
    # observation than no observation at all.
    is_usable &= peak >= np.percentile(peak[is_usable], 25)
    if is_usable.sum() < MIN_RIM_POINTS:
        return None

    edge_xy = (projected + offset[:, None] * normal)[is_usable]
    return visible_idx[is_usable], edge_xy, normal[is_usable]


def _fit_quality(point_count, residual_px):
    return {"n": int(point_count),
            "rms_px": float(np.sqrt((residual_px ** 2).mean())),
            "mad_px": float(np.median(np.abs(residual_px))),
            "inlier_frac_1px": float((np.abs(residual_px) < 1).mean())}


def fit_view(view_dir, topk=20, two_level=True, verbose=False,
             yaw_step=2.0, uv_step=0.010):
    view_dir = Path(view_dir)
    img = cv2.imread(str(view_dir / "color.png"))
    if img is None:
        return None
    img = cv2.undistort(img, K_4K.astype(np.float32), D_4K,
                        newCameraMatrix=K_4K.astype(np.float32))
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.6)
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)

    if two_level:
        P3, N2, lvl, loop = board_model.build_levels()
    else:
        p2, N2, loop = board_model.build()
        P3, lvl = board_model.pts3(p2), np.zeros(len(p2), int)

    cands, sinfo = seed_board.seed_view(view_dir, topk=topk, yaw_step=yaw_step,
                                        uv_step=uv_step)
    if not cands:
        return None

    best, binfo, bscore = None, None, np.inf
    for score, T in cands:
        rv, _ = cv2.Rodrigues(T[:3, :3])
        x0 = np.concatenate([rv.ravel(), T[:3, 3]])
        x, info = refine(gx, gy, P3, N2, lvl, loop, x0, two_level=two_level)
        if info is None:
            continue
        if verbose:
            print(f"      chamfer {score:5.2f} -> mad {info['mad_px']:5.2f} "
                  f"n {info['n']:4d}")
        # rank on the residual, not on the chamfer that produced the candidate
        if info["mad_px"] < bscore:
            bscore, best, binfo = info["mad_px"], x, info
    if best is None:
        return None

    R, t = RT(best)
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, t
    binfo.update(T_cam_board=T.tolist(), seed=sinfo,
                 n_table=sinfo.get("plane_normal"))
    return binfo


def _view_key(view_name):
    """(pose number, timestamp) for sorting and for matching finished rows."""
    pose_dir, timestamp = view_name.split("/")
    return int(pose_dir.split("_")[1]), timestamp


def _all_view_names(data):
    """Every pose_N/timestamp under `data`, in capture order."""
    names = []
    for pose_dir in data.glob("pose_*"):
        if not pose_dir.is_dir():
            continue
        for view_dir in sorted(pose_dir.iterdir()):
            if view_dir.is_dir():
                names.append(f"{pose_dir.name}/{view_dir.name}")
    return sorted(names, key=_view_key)


def _views_still_to_do(views, finished_rows):
    """Views with no row yet, so a killed run resumes instead of restarting."""
    done = set()
    for row in finished_rows:
        done.add((row["pose"], row["view"]))

    remaining = []
    for view_name in views:
        if _view_key(view_name) not in done:
            remaining.append(view_name)
    return remaining


def _residuals_for_pose(rows, pose):
    residuals = []
    for row in rows:
        if row["pose"] == pose:
            residuals.append(row["mad_px"])
    return residuals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("views", nargs="*", help="pose_N/timestamp; default all")
    ap.add_argument("--topk", type=int, default=20,
                    help="seed candidates refined per view")
    ap.add_argument("--top-only", action="store_true",
                    help="single-plane model (top rims only)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--restart", action="store_true",
                    help="ignore an existing --out file instead of resuming")
    ap.add_argument("--out", default=str(HERE / "out" / "rims6.json"))
    args = ap.parse_args()

    data = ROOT / "images" / "images_depth"
    views = args.views or _all_view_names(data)
    if args.limit:
        views = views[:args.limit]

    # Resume support. This run takes ~2 h; writing only at the end means a
    # power cut or a killed shell throws all of it away, which is exactly what
    # happened once. Results are flushed after every view and an existing
    # output file is reused, so a restart costs only the views not yet done.
    out_p = Path(args.out)
    out_p.parent.mkdir(exist_ok=True)
    rows = []
    if out_p.exists() and not args.restart:
        try:
            rows = json.load(open(out_p))
            views = _views_still_to_do(views, rows)
            print(f"resuming: {len(rows)} views already in {out_p.name}, "
                  f"{len(views)} to go")
        except (ValueError, KeyError):
            print(f"could not read {out_p.name}; starting over")
            rows = []

    t0 = time.time()
    for v in views:
        r = fit_view(data / v, topk=args.topk, two_level=not args.top_only,
                     verbose=args.verbose)
        if r is None:
            print(f"  {v:32} FAILED")
            continue
        p, vw = v.split("/")
        r.update(pose=int(p.split("_")[1]), view=vw, label="-")
        rows.append(r)
        json.dump(rows, open(out_p, "w"), indent=2)   # flush after every view
        print(f"  {v:32} mad {r['mad_px']:5.2f} px  rms {r['rms_px']:6.2f}  "
              f"in1px {r['inlier_frac_1px']:4.2f}  n {r['n']:4d}"
              f"{'   <-- PASS' if r['mad_px'] < 1.0 else ''}")

    if not rows:
        print("no views fitted")
        return
    m = np.array([r["mad_px"] for r in rows])
    print("\n" + "=" * 72)
    print(f"{len(rows)} views in {time.time()-t0:.0f}s   "
          f"mad median {np.median(m):.2f} px  p90 {np.percentile(m,90):.2f}  "
          f"pass<1.0 {(m<1.0).sum()}/{len(m)}")
    for p in sorted({r["pose"] for r in rows}):
        s = np.array(_residuals_for_pose(rows, p))
        print(f"  pose_{p:<3} n={len(s):3d}  median {np.median(s):5.2f}  "
              f"pass<1.0 {(s<1.0).sum():3d}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
