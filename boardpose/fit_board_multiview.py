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
from bundle_rims import (crop_bounds, gradient_images, measure,  # noqa: E402
                         project, project_with_normals)
from handeye_board import (average_pose, board_corners,  # noqa: E402
                           corner_spread_mm, se3, unse3)


# ------------------------------------------------------------------- loading

class MView:
    """Duck-types bundle_rims.View so measure() can be reused unchanged."""
    __slots__ = ("pose", "name", "gx", "gy", "ox", "oy", "rvec", "tvec",
                 "theta", "T_fk", "seed", "radius_px", "n_seed")


def load(pose, bundle, dtheta, data, max_radius, quiet=False):
    """Views of one folder that have BOTH a bundled board pose and joints."""
    bundled = json.load(open(bundle))
    intrinsics = np.array(bundled["intrinsics"], float)
    bias = board_model.pad_bias(bundled.get("rim_bias", board_model.RIM_BIAS0))
    bundled_views = _bundled_views_of_pose(bundled, pose)
    theta_of = _theta_by_view(pose)
    pts3 = board_model.build_levels()[0]

    views, rejections = [], []
    for (pose_id, view_name), bundled_view in sorted(bundled_views.items()):
        theta = theta_of.get((pose_id, view_name))
        if theta is None:
            continue
        image = _read_gray(data / f"pose_{pose_id}" / view_name / "color.png")
        if image is None:
            continue

        view, rejected_because = _build_view(pose_id, view_name, bundled_view,
                                             image, pts3, intrinsics, theta,
                                             dtheta, max_radius)
        if view is None:
            rejections.append((view_name, rejected_because))
            continue
        views.append(view)

    if not quiet:
        _print_rejections(rejections)
    return views, intrinsics, bias


def _bundled_views_of_pose(bundled, pose):
    """Bundled per-view records for one folder, keyed by (pose, view)."""
    selected = {}
    for record in bundled["views"]:
        if record["pose"] == pose:
            selected[(record["pose"], record["view"])] = record
    return selected


def _theta_by_view(pose):
    """Joint angles for one folder, keyed by (pose, view)."""
    theta_of = {}
    for pose_id, view_name, theta, _, _, _ in K.load_joint_views(poses=[pose]):
        theta_of[(pose_id, view_name)] = theta
    return theta_of


def _read_gray(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _print_rejections(rejections):
    for view_name, why in rejections:
        print(f"    dropped {view_name}: {why}")


def _board_offset_px(projected, intrinsics):
    """How far the board centre sits from the principal point."""
    return float(np.hypot(projected[:, 0].mean() - intrinsics[2],
                          projected[:, 1].mean() - intrinsics[3]))


def _build_view(pose_id, view_name, bundled_view, image, pts3, intrinsics,
                theta, dtheta, max_radius):
    """Returns (view, None), or (None, reason) when the view is unusable."""
    T_cam_board = np.asarray(bundled_view["T_cam_board"])
    rvec = cv2.Rodrigues(T_cam_board[:3, :3])[0].ravel()
    tvec = T_cam_board[:3, 3].copy()
    projected = project(pts3, rvec, tvec, intrinsics)

    # Tilt error explodes off axis: 2 of 147 views sit beyond 600 px from the
    # principal point and carry 1.45 deg of it, 8x the median, which mad_px
    # does not catch.
    offset_px = _board_offset_px(projected, intrinsics)
    if offset_px > max_radius:
        return None, f"board {offset_px:.0f} px off axis"

    x0, x1, y0, y1 = crop_bounds(projected, image.shape)
    if x1 - x0 < 200 or y1 - y0 < 200:
        return None, "board outside the frame"

    view = MView()
    view.pose, view.name = pose_id, view_name
    view.gx, view.gy = gradient_images(image, x0, x1, y0, y1)
    view.ox, view.oy = x0, y0
    view.rvec, view.tvec = rvec, tvec
    view.theta = theta
    view.T_fk = K.fk(theta + dtheta)
    view.seed = T_cam_board
    view.radius_px, view.n_seed = offset_px, int(bundled_view["n"])
    return view, None


# -------------------------------------------------------------------- the fit

DEFAULT_CHAIN_SIGMA = 0.40     # mm at the board; see the module docstring


def _chain_poses(views, X0):
    """Each view's own board pose pushed through the chain, in the FK frame."""
    return [view.T_fk @ X0 @ view.seed for view in views]


def seed_pose(views, X0):
    """Median of what the per-view poses already say, in the FK frame."""
    return average_pose(_chain_poses(views, X0))


def unpack(x, views, refine_x, X0):
    """(T_base_board, X, [per-view chain perturbation])."""
    T_base_board = se3(x[0:3], x[3:6])
    offset = 6

    X = X0
    if refine_x:
        X = se3(x[offset:offset + 3], x[offset + 3:offset + 6])
        offset += 6

    chain_slack = [x[offset + 6 * i:offset + 6 * i + 6]
                   for i in range(len(views))]
    return T_base_board, X, chain_slack


def poses_from(x, views, refine_x, X0):
    """Per-view T_cam_board implied by the current parameters."""
    T_base_board, X, chain_slack = unpack(x, views, refine_x, X0)
    X_inv = np.linalg.inv(X)
    return [_cam_board_pose(view, slack, X_inv, T_base_board)
            for view, slack in zip(views, chain_slack)]


def _cam_board_pose(view, slack, X_inv, T_base_board):
    """T_cam_board for one view under the current chain perturbation."""
    perturbed_chain = view.T_fk @ se3(slack[:3], slack[3:6])
    return X_inv @ np.linalg.inv(perturbed_chain) @ T_base_board


def apply_poses(x, views, refine_x, X0):
    for view, T_cam_board in zip(views, poses_from(x, views, refine_x, X0)):
        view.rvec = cv2.Rodrigues(T_cam_board[:3, :3])[0].ravel()
        view.tvec = T_cam_board[:3, 3]


def solve(views, intr, X0, pts3, nrm2, level, loop, bias=None, refine_x=False,
          chain_sigma=0.40, rim_sigma_px=0.63, schedule=(10, 6, 4, 4),
          seed_board=None, verbose=True):
    """Fit one shared T_base_board, with per-view chain slack under a prior.

    chain_sigma [mm] is the one knob: how much of a view's disagreement is
    blamed on the arm rather than on the board pose. rim_sigma_px only puts the
    image and prior residuals into the same units.
    """
    if bias is None:
        bias = board_model.RIM_BIAS0
    if seed_board is None:
        seed_board = seed_pose(views, X0)

    view_count = len(views)
    x = np.concatenate([unse3(seed_board)]
                       + ([unse3(X0)] if refine_x else [])
                       + [np.zeros(6 * view_count)])
    slack_offset = 6 + (6 if refine_x else 0)

    # Lever arm turning a chain rotation into a displacement at the board, so
    # the prior is one number in mm rather than separate mm and deg guesses.
    # Both residual blocks are carried in units of their own sigma, so they are
    # directly comparable and the robust loss means the same thing in each.
    lever_arm = np.array([np.linalg.norm(view.seed[:3, 3]) for view in views])

    def residuals(x, observations):
        T_base_board, X, chain_slack = unpack(x, views, refine_x, X0)
        X_inv = np.linalg.inv(X)
        blocks = []
        for view, slack, obs in zip(views, chain_slack, observations):
            if len(obs["idx"]) == 0:
                continue
            T_cam_board = _cam_board_pose(view, slack, X_inv, T_base_board)
            blocks.append(_rim_residual(T_cam_board, obs, pts3, nrm2, level,
                                        bias, intr) / rim_sigma_px)
        blocks.append(_chain_prior_residual(chain_slack, lever_arm, chain_sigma))
        return np.concatenate(blocks)

    def sparsity(observations):
        return _jacobian_sparsity(observations, view_count, slack_offset, len(x))

    for iteration, search_halfwidth in enumerate(schedule):
        apply_poses(x, views, refine_x, X0)
        observations = measure(views, intr, pts3, nrm2, search_halfwidth,
                               level, loop, bias)
        solution = least_squares(residuals, x,
                                 jac_sparsity=sparsity(observations),
                                 loss="soft_l1", f_scale=2.5, method="trf",
                                 max_nfev=60, args=(observations,))
        x = solution.x
        if verbose:
            _report_iteration(iteration, search_halfwidth, solution, views,
                              refine_x, X0, lever_arm, rim_sigma_px)

    apply_poses(x, views, refine_x, X0)
    T_base_board, X, chain_slack = unpack(x, views, refine_x, X0)
    return T_base_board, X, x, _chain_slack_mm(chain_slack, lever_arm)


def _rim_residual(T_cam_board, obs, pts3, nrm2, level, bias, intr):
    """Signed distance from each measured edge to the projected rim, in px."""
    idx = obs["idx"]
    biased = board_model.biased_pts(pts3[idx], nrm2[idx], level[idx],
                                    T_cam_board[:3, :3], T_cam_board[:3, 3], bias)
    projected, _ = project_with_normals(
        biased, nrm2[idx],
        cv2.Rodrigues(T_cam_board[:3, :3])[0].ravel(), T_cam_board[:3, 3], intr)
    return ((projected - obs["q"]) * obs["d"]).sum(1)


def _chain_slack_mm(chain_slack, lever_arm):
    """Each view's chain perturbation as one displacement at the board, mm."""
    return np.array([
        np.linalg.norm(np.concatenate([slack[:3] * lever, slack[3:6]])) * 1000
        for slack, lever in zip(chain_slack, lever_arm)])


def _chain_prior_residual(chain_slack, lever_arm, chain_sigma):
    """Gaussian prior: chain slack costs in units of chain_sigma.

    The reciprocal is formed once and multiplied, not divided per element --
    the two differ in the last bits and the fit is sensitive enough to show it.
    """
    per_sigma = 1.0 / chain_sigma
    scaled = [np.concatenate([slack[:3] * lever, slack[3:6]]) * 1000.0 * per_sigma
              for slack, lever in zip(chain_slack, lever_arm)]
    return np.concatenate(scaled)


def _jacobian_sparsity(observations, view_count, slack_offset, param_count):
    """Each view's rows touch the shared pose (and X) plus only its own slack."""
    row_count = sum(len(obs["idx"]) for obs in observations) + 6 * view_count
    pattern = lil_matrix((row_count, param_count), dtype=np.uint8)

    row = 0
    for i, obs in enumerate(observations):
        rows_here = len(obs["idx"])
        if rows_here == 0:
            continue
        pattern[row:row + rows_here, :slack_offset] = 1
        pattern[row:row + rows_here,
                slack_offset + 6 * i:slack_offset + 6 * i + 6] = 1
        row += rows_here

    for i in range(view_count):
        pattern[row + 6 * i:row + 6 * i + 6,
                slack_offset + 6 * i:slack_offset + 6 * i + 6] = 1
    return pattern


def _report_iteration(iteration, search_halfwidth, solution, views, refine_x,
                      X0, lever_arm, rim_sigma_px):
    rim_row_count = len(solution.fun) - 6 * len(views)
    residual_px = solution.fun[:rim_row_count] * rim_sigma_px
    _, _, chain_slack = unpack(solution.x, views, refine_x, X0)
    slack_mm = _chain_slack_mm(chain_slack, lever_arm)
    print(f"    iter {iteration} (S={search_halfwidth:2d})  "
          f"n={rim_row_count:6d}  "
          f"rms {np.sqrt((residual_px ** 2).mean()):.3f} px  "
          f"mad {np.median(np.abs(residual_px)):.3f} px  "
          f"in1px {(np.abs(residual_px) < 1).mean():.2f}   "
          f"chain slack median {np.median(slack_mm):.2f} mm")


# ------------------------------------------------------------------ reporting

def corner_delta(Ta, Tb):
    """RMS displacement of the four board corners between two poses, mm."""
    rms_mm, _ = corner_spread_mm(board_corners(Ta) - board_corners(Tb))
    return rms_mm


def pose_delta(Ta, Tb):
    """(corner rms mm, translation mm, rotation deg) between two poses."""
    dR = Ta[:3, :3].T @ Tb[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return (corner_delta(Ta, Tb),
            float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) * 1000), float(ang))


def average_per_view(views, X0):
    """The old way: push each view's own pose through the chain and average."""
    per_view = _chain_poses(views, X0)
    averaged = average_pose(per_view)
    scatter_mm = [corner_delta(T, averaged) for T in per_view]
    return averaged, np.array(scatter_mm)


def halves(views):
    """Split views into two sets that interleave in capture order, so both
    halves span the same range of arm configurations."""
    idx = np.argsort([v.name for v in views])
    a = [views[i] for i in idx[0::2]]
    b = [views[i] for i in idx[1::2]]
    return a, b


# ------------------------------------------------------------------------ CLI

def _hand_eye_is_identifiable(views, min_rotation_deg=15.0):
    """X is only identifiable if the wrist actually rotated between views."""
    rotations_deg = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            relative = views[i].T_fk[:3, :3].T @ views[j].T_fk[:3, :3]
            rotations_deg.append(np.degrees(np.arccos(
                np.clip((np.trace(relative) - 1) / 2, -1, 1))))

    widest_deg = max(rotations_deg)
    if widest_deg < min_rotation_deg:
        print(f"  refusing --refine-x: max relative rotation between views "
              f"is {widest_deg:.1f} deg, so X is not identifiable here. "
              f"Using the calibrated X.")
        return False

    print(f"  --refine-x: max relative rotation {widest_deg:.1f} deg, "
          f"X is identifiable")
    return True


def _report_per_view_residuals(views, intr, pts3, nrm2, level, loop, bias):
    """How well each view is served by the single shared pose."""
    observations = measure(views, intr, pts3, nrm2, 4, level, loop, bias)
    print("\n  per-view rim residual under the shared pose:")
    for view, obs in zip(views, observations):
        biased = board_model.biased_pts(
            pts3[obs["idx"]], nrm2[obs["idx"]], level[obs["idx"]],
            cv2.Rodrigues(view.rvec)[0], view.tvec, bias)
        projected, _ = project_with_normals(biased, nrm2[obs["idx"]],
                                            view.rvec, view.tvec, intr)
        residual_px = ((projected - obs["q"]) * obs["d"]).sum(1)
        print(f"    {view.name:20} n {len(residual_px):5d}  "
              f"mad {np.median(np.abs(residual_px)):5.3f} px"
              f"  in1px {(np.abs(residual_px) < 1).mean():4.2f}  "
              f"({view.radius_px:4.0f} px off axis)")


def _print_board_pose(T_base_board):
    print("T_base_board =")
    for row in T_base_board:
        print("  [" + "  ".join(f"{value:+12.6f}" for value in row) + "]")
    translation_mm = T_base_board[:3, 3] * 1000
    rotation_deg = np.degrees(cv2.Rodrigues(T_base_board[:3, :3])[0].ravel())
    print(f"  translation [{translation_mm[0]:+9.3f} {translation_mm[1]:+9.3f} "
          f"{translation_mm[2]:+9.3f}] mm")
    print(f"  rotation    [{rotation_deg[0]:+9.3f} {rotation_deg[1]:+9.3f} "
          f"{rotation_deg[2]:+9.3f}] deg")


def _run_split_half(views, intr, X0, pts3, nrm2, level, loop, bias,
                    refine_x, chain_sigma):
    """Fit two disjoint halves independently and compare. No ground truth."""
    print("\n" + "=" * 72)
    print("split-half validation: two disjoint halves of the views, fitted")
    print("independently. No ground truth is involved.")
    half_a, half_b = halves(views)

    print(f"\n  half A ({len(half_a)} views)")
    T_a, _, _, _ = solve(half_a, intr, X0, pts3, nrm2, level, loop, bias,
                         refine_x=refine_x, chain_sigma=chain_sigma,
                         verbose=False)
    print(f"  half B ({len(half_b)} views)")
    T_b, _, _, _ = solve(half_b, intr, X0, pts3, nrm2, level, loop, bias,
                         refine_x=refine_x, chain_sigma=chain_sigma,
                         verbose=False)
    joint = pose_delta(T_a, T_b)

    averaged_a, _ = average_per_view(half_a, X0)
    averaged_b, _ = average_per_view(half_b, X0)
    averaged = pose_delta(averaged_a, averaged_b)

    print("\n  half-vs-half disagreement (2x the precision of each half):")
    print(f"    averaged per-view poses : {averaged[0]:6.3f} mm at the corner   "
          f"({averaged[1]:.3f} mm, {averaged[2]:.4f} deg)")
    print(f"    joint multi-view fit    : {joint[0]:6.3f} mm at the corner   "
          f"({joint[1]:.3f} mm, {joint[2]:.4f} deg)")
    if joint[0] > 0:
        print(f"    -> {averaged[0] / joint[0]:.2f}x better")

    return {"joint_fit_mm": joint[0], "avg_per_view_mm": averaged[0],
            "joint_fit_trans_mm": joint[1], "joint_fit_rot_deg": joint[2],
            "avg_trans_mm": averaged[1], "avg_rot_deg": averaged[2]}


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
        args.refine_x = _hand_eye_is_identifiable(views)

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

    _report_per_view_residuals(views, intr, pts3, nrm2, level, loop, bias)

    # put the base mounting transform back, so the answer is in the frame the
    # folder's own pose.txt uses
    jv = [v for v in K.load_joint_views(poses=[args.pose])]
    Mb, sc = K.fit_base_transform(jv)
    T_base_board = Mb @ Tb
    frame = jv[0][5] if jv else "?"
    print(f"\nbase mounting transform recovered from this folder "
          f"(scatter {np.median(sc):.3f} mm), frame '{frame}'")
    _print_board_pose(T_base_board)

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
        result["split_half"] = _run_split_half(
            views, intr, X0, pts3, nrm2, level, loop, bias,
            args.refine_x, chain_sigma)

    out = args.out or str(HERE / "out" / f"board_multiview_pose_{args.pose}.json")
    json.dump(result, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
