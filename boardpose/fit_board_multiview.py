#!/usr/bin/env python3
"""
fit_board_multiview.py - ONE board pose in the robot frame from N shots.

The board is static within a folder and the arm moves around it, so N captures
are N looks at the same rigid unknown. This fits that unknown directly, from
every view's subpixel rim edges at once:

    T_cam_board_i  =  X^-1 @ exp(-xi_i) @ FK(theta_i + dtheta)^-1 @ T_base_board

THE MODEL THAT MATTERS IS xi_i

The tempting version of this fit drops xi_i and ties every view rigidly to one
board pose. Do not: it is measurably WORSE than the pose averaging it replaces
(pose_12 split-half, 0.788 mm vs 0.524 mm at the corner). The reason is the
error budget, and it is worth stating plainly:

    vision, per view, in the camera frame       0.13 mm  (rim mad 0.63 px)
    the chain, per view, after joint offsets    0.87 mm

The chain is seven times noisier than the vision. A rigid fit asserts the chain
is exact, so all 0.87 mm of chain error has nowhere to go but into the board
pose -- weighted by rim-point count, which is not remotely the right weight for
it. Forcing one rigid pose costs 0.39 px of image fit (0.63 -> 1.02 px mad),
and that misfit is the chain being wrong, not the board moving.

So each view gets its own 6-DoF chain perturbation xi_i, penalised by a
Gaussian prior of width --chain-sigma. The board pose stays shared. That model
contains both alternatives as limits:

    chain_sigma -> 0     rigid fit, the chain is perfect
    chain_sigma -> inf   independent per-view fits, and the shared pose becomes
                         their information-weighted combination

and the honest answer sits in between. eval_multiview.py --sweep scans it over
all 11 folders; 0.40 mm is the default because it won there. Note that is BELOW
the 0.87 mm residual kinematic_calib.py reports, and it should be: that figure
is a total, including the vision and board-pose error, and within a single
folder the common-mode part of the chain error is absorbed by the shared board
pose anyway. Only the view-to-view part has to be modelled here.

What the shared pose buys over plain averaging is
that the combination is weighted by the information that actually constrains
each direction. A single view measures the board ANISOTROPICALLY -- split-half
precision is 0.012 deg in yaw against 0.061 deg in tilt, because tilt is seen
only through the 22 mm baseline between the pocket floors and the top rims.
Averaging poses treats those as equally trustworthy; this does not. A view with
664 visible rim points instead of 1147 gets its due weight, and `soft_l1` over
pooled observations downweights bad EDGES rather than whole views.

MEASURED, over all 11 folders (eval_multiview.py, split-half over the views --
each half fitted independently, the board never moved, nothing external
involved):

                        joint fit   averaging
    median                0.314 mm    0.389 mm
    mean                  0.323       0.490
    worst folder          0.554       1.441

7 of 11 folders improve. Read that shape carefully, because it is the honest
description: this is a REGULARISED estimator. Where averaging is already good it
is slightly worse (pose_9 0.218 vs 0.148); where averaging is bad it is much
better (pose_7 0.448 vs 1.441). The mean and the worst case are what move, not
the best case. For a production estimator that cannot know in advance which
folder it is in, the worst case is the number that matters.

CAVEAT: chain_sigma was chosen on this same split-half statistic, so the 1.24x
at the median is an upper bound on it. The gain does not depend on that choice
-- every value tested from 0.4 to 8.0 mm beats averaging, at 1.07x to 1.24x --
but do not quote 1.24x as an independently validated figure.

WHAT IS HELD FIXED, AND WHY

  intrinsics   from the bundle (bundle_rims.py), which calibrated fx..k3 on the
               board itself over all 147 views. Refitting 9 intrinsics against
               one folder would just re-absorb the board pose.
  rim bias     also from the bundle. The STL outline is not where the intensity
               edge is, and because tilt is triangulated from the 22 mm baseline
               between the two contour levels, the differential bias between
               them is a tilt error by construction -- it was the whole 0.182
               deg per-view tilt error. See board_model.rim_bias().
  dtheta       joint offsets from kinematic_calib.py. They are a property of
               the arm, cross-validated 11/11 folds, and must NOT be refit here
               -- against a single folder they are perfectly degenerate with
               the board pose they are supposed to correct.
  X            hand-eye, from kinematic_calib.py. Refittable with --refine-x if
               the camera stick was disturbed since that calibration; it is
               identifiable per folder provided the wrist actually rotated
               between views, which `--refine-x` checks before allowing it.

The fit runs in the FK frame (kinematics.fk(theta), no base mounting transform), for
the reason kinematic_calib.py gives: the mounting transform is a constant
left-multiplication, absorbed exactly by the board pose. It is put back at the
end, from this folder's own captures, so the reported pose is in the frame the
folder's pose.txt uses.
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
import kinematics as K  # noqa: E402
from bundle_rims import MARGIN, measure, project, project_with_normals  # noqa: E402
from handeye_board import CORNERS, se3, unse3  # noqa: E402


# ------------------------------------------------------------------- loading

class MView:
    """Duck-types bundle_rims.View so measure() can be reused unchanged."""
    __slots__ = ("pose", "name", "gx", "gy", "ox", "oy", "rvec", "tvec",
                 "theta", "T_fk", "seed", "radius_px", "n_seed")


def load(pose, bundle, dtheta, data, max_radius, quiet=False):
    """Views of one folder that have BOTH a bundled board pose and joints."""
    B = json.load(open(bundle))
    intr = np.array(B["intrinsics"], float)
    bias = board_model.pad_bias(B.get("rim_bias", board_model.RIM_BIAS0))
    seed = {(r["pose"], r["view"]): r for r in B["views"] if r["pose"] == pose}
    joints = {(p, v): th for p, v, th, _, _, _ in K.load_joint_views(poses=[pose])}

    pts3 = board_model.build_levels()[0]
    out, dropped = [], []
    for (p, v), r in sorted(seed.items()):
        if (p, v) not in joints:
            continue
        img = cv2.imread(str(data / f"pose_{p}" / v / "color.png"),
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        T = np.asarray(r["T_cam_board"])
        rvec = cv2.Rodrigues(T[:3, :3])[0].ravel()
        tvec = T[:3, 3].copy()

        c = project(pts3, rvec, tvec, intr)
        # Gate on where the board sits in the frame. Tilt error explodes off
        # axis -- 2 of 147 views sit beyond 600 px from the principal point and
        # carry 1.45 deg of it, 8x the median, and mad_px does not catch them.
        rad = float(np.hypot(c[:, 0].mean() - intr[2], c[:, 1].mean() - intr[3]))
        if rad > max_radius:
            dropped.append((v, f"board {rad:.0f} px off axis"))
            continue
        x0 = int(max(0, c[:, 0].min() - MARGIN))
        x1 = int(min(img.shape[1], c[:, 0].max() + MARGIN))
        y0 = int(max(0, c[:, 1].min() - MARGIN))
        y1 = int(min(img.shape[0], c[:, 1].max() + MARGIN))
        if x1 - x0 < 200 or y1 - y0 < 200:
            dropped.append((v, "board outside the frame"))
            continue

        g = cv2.GaussianBlur(img[y0:y1, x0:x1], (0, 0), 1.6)
        m = MView()
        m.pose, m.name = p, v
        m.gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
        m.gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
        m.ox, m.oy = x0, y0
        m.rvec, m.tvec = rvec, tvec
        m.theta = joints[(p, v)]
        m.T_fk = K.fk(m.theta + dtheta)
        m.seed = T
        m.radius_px, m.n_seed = rad, int(r["n"])
        out.append(m)

    if dropped and not quiet:
        for v, why in dropped:
            print(f"    dropped {v}: {why}")
    return out, intr, bias


# -------------------------------------------------------------------- the fit

DEFAULT_CHAIN_SIGMA = 0.40     # mm at the board; see the module docstring


def seed_pose(views, X0):
    """Median of what the per-view poses already say, in the FK frame."""
    Ts = [v.T_fk @ X0 @ v.seed for v in views]
    S = np.eye(4)
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    S[:3, :3] = U @ Vt
    if np.linalg.det(S[:3, :3]) < 0:
        U[:, -1] *= -1; S[:3, :3] = U @ Vt
    S[:3, 3] = np.median([T[:3, 3] for T in Ts], axis=0)
    return S


def unpack(x, views, refine_x, X0):
    """(T_base_board, X, [per-view chain perturbation])."""
    Tb = se3(x[0:3], x[3:6])
    o = 6
    X = X0
    if refine_x:
        X = se3(x[o:o + 3], x[o + 3:o + 6]); o += 6
    xi = [x[o + 6 * i:o + 6 * i + 6] for i in range(len(views))]
    return Tb, X, xi


def poses_from(x, views, refine_x, X0):
    """Per-view T_cam_board implied by the current parameters."""
    Tb, X, xi = unpack(x, views, refine_x, X0)
    Xi = np.linalg.inv(X)
    return [Xi @ np.linalg.inv(v.T_fk @ se3(z[:3], z[3:6])) @ Tb
            for v, z in zip(views, xi)]


def apply_poses(x, views, refine_x, X0):
    for v, T in zip(views, poses_from(x, views, refine_x, X0)):
        v.rvec = cv2.Rodrigues(T[:3, :3])[0].ravel()
        v.tvec = T[:3, 3]


def solve(views, intr, X0, pts3, nrm2, level, loop, bias=None, refine_x=False,
          chain_sigma=0.40, rim_sigma_px=0.63, schedule=(10, 6, 4, 4),
          seed_board=None, verbose=True):
    """Fit one shared T_base_board, with per-view chain slack under a prior.

    chain_sigma  [mm]  how far the chain is trusted, as a displacement at the
                       board. This is the single knob that decides how much a
                       view's disagreement is blamed on the arm rather than on
                       the board pose.
    rim_sigma_px [px]  the rim edge noise, used only to put the image residuals
                       and the prior residuals into the same units.
    """
    if bias is None:
        bias = board_model.RIM_BIAS0
    if seed_board is None:
        seed_board = seed_pose(views, X0)
    nv = len(views)
    x = np.concatenate([unse3(seed_board)]
                       + ([unse3(X0)] if refine_x else [])
                       + [np.zeros(6 * nv)])
    o_xi = 6 + (6 if refine_x else 0)

    # Lever arm turning a chain rotation into a displacement at the board, so
    # the prior is one number in mm rather than separate mm and deg guesses.
    # Both residual blocks are carried in units of their own sigma, so they are
    # directly comparable and the robust loss means the same thing in each.
    lever = np.array([np.linalg.norm(v.seed[:3, 3]) for v in views])
    w_prior = 1.0 / chain_sigma          # mm of board displacement -> sigmas

    def residuals(x, obs):
        Tb, X, xi = unpack(x, views, refine_x, X0)
        Xi = np.linalg.inv(X)
        out = []
        for i, (v, o) in enumerate(zip(views, obs)):
            if len(o["idx"]) == 0:
                continue
            T = Xi @ np.linalg.inv(v.T_fk @ se3(xi[i][:3], xi[i][3:6])) @ Tb
            idx = o["idx"]
            P = board_model.biased_pts(pts3[idx], nrm2[idx], level[idx],
                                       T[:3, :3], T[:3, 3], bias)
            a, _ = project_with_normals(P, nrm2[idx],
                                        cv2.Rodrigues(T[:3, :3])[0].ravel(),
                                        T[:3, 3], intr)
            out.append(((a - o["q"]) * o["d"]).sum(1) / rim_sigma_px)
        # prior: chain slack costs, in the same units, as a board displacement
        pri = np.concatenate([
            np.concatenate([xi[i][:3] * lever[i], xi[i][3:6]]) * 1000.0 * w_prior
            for i in range(nv)])
        out.append(pri)
        return np.concatenate(out)

    def sparsity(obs):
        n = sum(len(o["idx"]) for o in obs) + 6 * nv
        J = lil_matrix((n, len(x)), dtype=np.uint8)
        r = 0
        for i, o in enumerate(obs):
            k = len(o["idx"])
            if k == 0:
                continue
            J[r:r + k, :o_xi] = 1                       # board pose (and X)
            J[r:r + k, o_xi + 6 * i:o_xi + 6 * i + 6] = 1
            r += k
        for i in range(nv):                             # prior rows
            J[r + 6 * i:r + 6 * i + 6, o_xi + 6 * i:o_xi + 6 * i + 6] = 1
        return J

    sol = None
    for it, S in enumerate(schedule):
        apply_poses(x, views, refine_x, X0)
        obs = measure(views, intr, pts3, nrm2, S, level, loop, bias)
        sol = least_squares(residuals, x, jac_sparsity=sparsity(obs),
                            loss="soft_l1", f_scale=2.5, method="trf",
                            max_nfev=60, args=(obs,))
        x = sol.x
        if verbose:
            nrim = len(sol.fun) - 6 * nv
            r = sol.fun[:nrim] * rim_sigma_px
            _, _, xi = unpack(x, views, refine_x, X0)
            slack = [np.linalg.norm(np.concatenate(
                [z[:3] * lever[i], z[3:6]])) * 1000 for i, z in enumerate(xi)]
            print(f"    iter {it} (S={S:2d})  n={nrim:6d}  "
                  f"rms {np.sqrt((r ** 2).mean()):.3f} px  "
                  f"mad {np.median(np.abs(r)):.3f} px  "
                  f"in1px {(np.abs(r) < 1).mean():.2f}   "
                  f"chain slack median {np.median(slack):.2f} mm")

    apply_poses(x, views, refine_x, X0)
    Tb, X, xi = unpack(x, views, refine_x, X0)
    slack = np.array([np.linalg.norm(np.concatenate(
        [z[:3] * lever[i], z[3:6]])) * 1000 for i, z in enumerate(xi)])
    return Tb, X, x, slack


# ------------------------------------------------------------------ reporting

def corner_delta(Ta, Tb):
    """RMS displacement of the four board corners between two poses, mm."""
    a = CORNERS @ Ta[:3, :3].T + Ta[:3, 3]
    b = CORNERS @ Tb[:3, :3].T + Tb[:3, 3]
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()) * 1000)


def pose_delta(Ta, Tb):
    """(corner rms mm, translation mm, rotation deg) between two poses."""
    dR = Ta[:3, :3].T @ Tb[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return (corner_delta(Ta, Tb),
            float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) * 1000), float(ang))


def average_per_view(views, X0):
    """The old way: push each view's own pose through the chain and average."""
    Ts = [v.T_fk @ X0 @ v.seed for v in views]
    M = np.eye(4)
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    M[:3, :3] = U @ Vt
    if np.linalg.det(M[:3, :3]) < 0:
        U[:, -1] *= -1; M[:3, :3] = U @ Vt
    M[:3, 3] = np.median([T[:3, 3] for T in Ts], axis=0)
    scatter = [corner_delta(T, M) for T in Ts]
    return M, np.array(scatter)


def halves(views):
    """Split views into two sets that interleave in capture order, so both
    halves span the same range of arm configurations."""
    idx = np.argsort([v.name for v in views])
    a = [views[i] for i in idx[0::2]]
    b = [views[i] for i in idx[1::2]]
    return a, b


# ------------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", type=int, required=True, help="pose_N folder")
    ap.add_argument("--bundle", default=str(HERE / "out" / "bundle_rims6_v3.json"))
    ap.add_argument("--calib", default=str(HERE / "out" / "kinematic_calib_v3.json"),
                    help="kinematic_calib.py output: supplies dtheta and X")
    ap.add_argument("--model", default="joint offsets only",
                    help="which kinematic_calib model to take dtheta/X from")
    ap.add_argument("--chain-sigma", type=float, default=None,
                    help="how far the robot chain is trusted, as a board "
                         "displacement [mm]. Default 0.40, which won the "
                         "eval_multiview.py --sweep over all 11 folders.")
    ap.add_argument("--refine-x", action="store_true",
                    help="also refit the hand-eye against this folder")
    ap.add_argument("--max-radius", type=float, default=500.0,
                    help="drop views whose board centre is further than this "
                         "from the principal point [px]")
    ap.add_argument("--split-half", action="store_true",
                    help="fit two disjoint halves of the views and compare "
                         "-- the honest precision, no ground truth involved")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cal = json.load(open(args.calib))[args.model]
    dtheta = np.radians(cal["dtheta_deg"])
    X0 = np.asarray(cal["X_tcp_cam"])
    print(f"kinematic model '{args.model}':  dtheta (deg) "
          + " ".join(f"{d:+.3f}" for d in cal["dtheta_deg"])
          + f"   |t_X| {np.linalg.norm(X0[:3, 3]) * 1000:.1f} mm")

    chain_sigma = args.chain_sigma or DEFAULT_CHAIN_SIGMA
    print(f"chain trusted to {chain_sigma:.3f} mm at the board "
          + ("(--chain-sigma)" if args.chain_sigma else "(default)")
          + f";  kinematic_calib reports {cal.get('median_mm', float('nan')):.3f} mm "
            "overall, which includes vision and board-pose error")

    pts3, nrm2, level, loop = board_model.build_levels()
    data = ROOT / "images" / "images_depth"
    t0 = time.time()
    views, intr, bias = load(args.pose, args.bundle, dtheta, data, args.max_radius)
    if len(views) < 2:
        print(f"pose_{args.pose}: {len(views)} usable views -- need at least 2")
        return
    print(f"pose_{args.pose}: {len(views)} views, {len(pts3)} rim points each "
          f"({time.time() - t0:.0f}s to load)")

    if args.refine_x:
        # X is only identifiable if the wrist actually rotated between views.
        ang = []
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                dR = views[i].T_fk[:3, :3].T @ views[j].T_fk[:3, :3]
                ang.append(np.degrees(np.arccos(
                    np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        if max(ang) < 15.0:
            print(f"  refusing --refine-x: max relative rotation between views "
                  f"is {max(ang):.1f} deg, so X is not identifiable here. "
                  f"Using the calibrated X.")
            args.refine_x = False
        else:
            print(f"  --refine-x: max relative rotation {max(ang):.1f} deg, "
                  f"X is identifiable")

    # the old way, for comparison
    M_avg, scatter = average_per_view(views, X0)
    print(f"\nper-view poses pushed through the chain and averaged (the old way):"
          f"\n  cross-view scatter at the corner: median {np.median(scatter):.3f}"
          f"  p90 {np.percentile(scatter, 90):.3f}  max {scatter.max():.3f} mm")

    print("\njoint fit over every view's rim edges:")
    Tb, X, x, slack = solve(views, intr, X0, pts3, nrm2, level, loop, bias,
                            refine_x=args.refine_x, chain_sigma=chain_sigma)
    print(f"  per-view chain slack absorbed: median {np.median(slack):.3f}  "
          f"p90 {np.percentile(slack, 90):.3f}  max {slack.max():.3f} mm")

    d = pose_delta(M_avg, Tb)
    print(f"\n  joint fit vs averaged per-view poses: {d[0]:.3f} mm at the "
          f"corner  ({d[1]:.3f} mm translation, {d[2]:.4f} deg rotation)")

    # residual of each view under the single shared pose
    obs = measure(views, intr, pts3, nrm2, 4, level, loop, bias)
    print("\n  per-view rim residual under the shared pose:")
    for v, o in zip(views, obs):
        P = board_model.biased_pts(pts3[o["idx"]], nrm2[o["idx"]], level[o["idx"]],
                                   cv2.Rodrigues(v.rvec)[0], v.tvec, bias)
        a, _ = project_with_normals(P, nrm2[o["idx"]], v.rvec, v.tvec, intr)
        r = ((a - o["q"]) * o["d"]).sum(1)
        print(f"    {v.name:20} n {len(r):5d}  mad {np.median(np.abs(r)):5.3f} px"
              f"  in1px {(np.abs(r) < 1).mean():4.2f}  ({v.radius_px:4.0f} px off axis)")

    # put the base mounting transform back, so the answer is in the frame the
    # folder's own pose.txt uses
    jv = [v for v in K.load_joint_views(poses=[args.pose])]
    Mb, sc = K.fit_base_transform(jv)
    T_base_board = Mb @ Tb
    frame = jv[0][5] if jv else "?"
    print(f"\nbase mounting transform recovered from this folder "
          f"(scatter {np.median(sc):.3f} mm), frame '{frame}'")
    print("T_base_board =")
    for row in T_base_board:
        print("  [" + "  ".join(f"{c:+12.6f}" for c in row) + "]")
    t = T_base_board[:3, 3] * 1000
    rv = np.degrees(cv2.Rodrigues(T_base_board[:3, :3])[0].ravel())
    print(f"  translation [{t[0]:+9.3f} {t[1]:+9.3f} {t[2]:+9.3f}] mm")
    print(f"  rotation    [{rv[0]:+9.3f} {rv[1]:+9.3f} {rv[2]:+9.3f}] deg")

    result = {"pose": args.pose, "n_views": len(views),
              "views": [v.name for v in views],
              "frame": frame,
              "T_base_board": T_base_board.tolist(),
              "T_fkbase_board": Tb.tolist(),
              "X_tcp_cam": X.tolist(),
              "refined_x": bool(args.refine_x),
              "chain_sigma_mm": chain_sigma,
              "chain_slack_mm": list(map(float, slack)),
              "avg_per_view": {"T_base_board": (Mb @ M_avg).tolist(),
                               "scatter_mm": list(map(float, scatter))}}

    if args.split_half:
        print("\n" + "=" * 72)
        print("split-half validation: two disjoint halves of the views, fitted")
        print("independently. No ground truth is involved.")
        A, Bv = halves(views)
        print(f"\n  half A ({len(A)} views)")
        Ta, _, _, _ = solve(A, intr, X0, pts3, nrm2, level, loop, bias,
                            refine_x=args.refine_x, chain_sigma=chain_sigma,
                            verbose=False)
        print(f"  half B ({len(Bv)} views)")
        Tbb, _, _, _ = solve(Bv, intr, X0, pts3, nrm2, level, loop, bias,
                             refine_x=args.refine_x, chain_sigma=chain_sigma,
                             verbose=False)
        dj = pose_delta(Ta, Tbb)

        Ma, _ = average_per_view(A, X0)
        Mbv, _ = average_per_view(Bv, X0)
        da = pose_delta(Ma, Mbv)

        print(f"\n  half-vs-half disagreement (2x the precision of each half):")
        print(f"    averaged per-view poses : {da[0]:6.3f} mm at the corner   "
              f"({da[1]:.3f} mm, {da[2]:.4f} deg)")
        print(f"    joint multi-view fit    : {dj[0]:6.3f} mm at the corner   "
              f"({dj[1]:.3f} mm, {dj[2]:.4f} deg)")
        if dj[0] > 0:
            print(f"    -> {da[0] / dj[0]:.2f}x better")
        result["split_half"] = {"joint_fit_mm": dj[0],
                                "avg_per_view_mm": da[0],
                                "joint_fit_trans_mm": dj[1],
                                "joint_fit_rot_deg": dj[2],
                                "avg_trans_mm": da[1], "avg_rot_deg": da[2]}

    out = args.out or str(HERE / "out" / f"board_multiview_pose_{args.pose}.json")
    json.dump(result, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
