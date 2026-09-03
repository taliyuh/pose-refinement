#!/usr/bin/env python3
"""
board_scatter.py - the metric that actually matters, and where it comes from.

The rim residual (mad_px, ~0.14 mm) says the projected model lands on the image
edges. It is a measure of FIT, not of POSE: it is a 1-D distance along the edge
normal, in the camera frame, for one view at a time. It cannot see an error
that moves the board along the viewing ray, and it says nothing about whether
two views of the SAME static board agree.

This measures that instead. Within one folder the board does not move, so

    T_base_board(i) = T_base_tcp(i) @ X @ T_cam_board(i)

must be identical for every view i. Its scatter over the views is an
end-to-end error: board fit + hand-eye + robot FK, in millimetres, in the frame
the user cares about. That is the number to report.

It also classifies the outliers. A chamfer fit on a near-periodic array of
pockets has two failure modes that both score well: a yaw flip (the board is
roughly 2-fold symmetric) and a lattice slip (the fit lands one pocket pitch
off). Both are detectable as a large deviation from the folder consensus with a
characteristic signature, so they can be rejected rather than averaged in.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import kinematics as K  # noqa: E402
from handeye_board import (average_pose, board_corners,  # noqa: E402
                           calibrate_hand_eye_park, corner_spread_mm,
                           se3, unse3)
from scipy.optimize import least_squares  # noqa: E402

MOUNT_A = [1, 2, 3]          # camera mount before the pose_3/pose_4 remount


def fit_x_and_boards(Tb, Tc, grp, huber=0.003, w=None):
    """Shared hand-eye X + one board pose per folder, by robust least squares.

    The Park & Martin closed form is only a seed. Used on its own it leaves
    ~20 mm of folder scatter here: it solves rotation and translation in two
    decoupled stages and is not robust, so a few bad views drag it. Refining X
    and the board poses together against the corner residual is what makes the
    number meaningful.
    """
    folders = sorted(set(grp))
    fidx = {p: i for i, p in enumerate(folders)}
    gi = np.array([fidx[p] for p in grp])
    w = np.ones(len(Tb)) if w is None else np.asarray(w, float)

    R_x, t_x = calibrate_hand_eye_park([T[:3, :3] for T in Tb],
                                       [T[:3, 3] for T in Tb],
                                       [T[:3, :3] for T in Tc],
                                       [T[:3, 3] for T in Tc])
    X0 = np.eye(4); X0[:3, :3] = R_x; X0[:3, 3] = t_x.ravel()

    seeds = []
    for folder in folders:
        predicted = []
        for i in range(len(Tb)):
            if gi[i] == fidx[folder]:
                predicted.append(Tb[i] @ X0 @ Tc[i])
        seeds.append(unse3(consensus(predicted)))
    x0 = np.concatenate([unse3(X0)] + seeds)
    tcp_poses = np.array(Tb)

    def resid(x):
        X = se3(x[0:3], x[3:6])
        board_poses = _unpack_board_poses(x, len(folders))
        corner_errors = np.empty((len(Tb), 12))
        for i in range(len(Tb)):
            predicted = tcp_poses[i] @ X @ Tc[i]
            delta = (board_corners(predicted)
                     - board_corners(board_poses[gi[i]]))
            corner_errors[i] = w[i] * delta.ravel()
        return corner_errors.ravel()

    solution = least_squares(resid, x0, loss="soft_l1", f_scale=huber,
                             max_nfev=300)
    X = se3(solution.x[0:3], solution.x[3:6])
    board_poses = _unpack_board_poses(solution.x, len(folders))

    boards = {}
    for folder, i in fidx.items():
        boards[folder] = board_poses[i]
    return X, boards


def _unpack_board_poses(x, folder_count):
    """The per-folder board poses packed after the hand-eye."""
    board_poses = []
    for i in range(folder_count):
        board_poses.append(se3(x[6 + 6 * i:9 + 6 * i], x[9 + 6 * i:12 + 6 * i]))
    return board_poses


# consensus() is the shared rotation-average; kept as a local name because
# "consensus of a folder" is what it means here.
consensus = average_pose


def corner_dev(pose, consensus_pose):
    """RMS deviation of `pose` from the consensus, over the board corners, mm.

    Corners rather than the origin: an origin-only distance is blind to a pure
    rotation about the origin, which is exactly what a yaw flip is.
    """
    delta = board_corners(pose) - board_corners(consensus_pose)
    rms_mm, _ = corner_spread_mm(delta)
    return rms_mm


def yaw_about_normal(pose, consensus_pose):
    """Rotation of `pose` relative to consensus, as (angle deg, |axis.z|)."""
    relative = consensus_pose[:3, :3].T @ pose[:3, :3]
    angle_deg = np.degrees(np.arccos(
        np.clip((np.trace(relative) - 1) / 2, -1, 1)))

    # The axis component along board z separates a yaw flip from a tilt error.
    axis = np.array([relative[2, 1] - relative[1, 2],
                     relative[0, 2] - relative[2, 0],
                     relative[1, 0] - relative[0, 1]])
    axis_length = np.linalg.norm(axis)
    if axis_length <= 1e-9:
        return angle_deg, 1.0
    return angle_deg, abs(axis[2] / axis_length)


@dataclass
class _Capture:
    """One view with everything the hand-eye fit needs from it."""
    pose: int
    view: str
    theta: object            # None for legacy cached captures
    T_base_tcp: np.ndarray
    T_cam_board: np.ndarray
    mad_px: object


def board_poses(src, cache_p=None, use_offsets=None, gate=1.2):
    """T_base_board per view, grouped by folder, with a per-mount hand-eye X."""
    bundled = json.load(open(src))
    records = bundled["views"] if isinstance(bundled, dict) else bundled
    board_of, mad_of = _index_bundle(records)

    captures = _captures_with_joints(board_of, mad_of)
    captures += _captures_from_cache(cache_p, board_of, mad_of, captures)
    if use_offsets is not None:
        _apply_joint_offsets(captures, use_offsets)

    poses_by_folder, fitted_boards = {}, {}
    for mount in ("A", "B"):
        on_mount = _captures_on_mount(captures, mount)
        if not on_mount:
            continue
        X = _fit_mount(on_mount, gate, fitted_boards)
        for capture in on_mount:
            poses_by_folder.setdefault(capture.pose, []).append(
                (capture.view, capture.T_base_tcp @ X @ capture.T_cam_board,
                 capture.mad_px, capture.T_cam_board))
    return poses_by_folder, fitted_boards


def _index_bundle(records):
    """(T_cam_board, mad_px) keyed by (pose, view)."""
    board_of, mad_of = {}, {}
    for record in records:
        key = (record["pose"], record["view"])
        board_of[key] = np.asarray(record["T_cam_board"])
        mad_of[key] = record.get("mad_px")
    return board_of, mad_of


def _captures_with_joints(board_of, mad_of):
    """T_base_tcp comes from the joint angles, so offsets can be applied."""
    captures = []
    for pose, view, theta, T_base_tcp, _, _ in K.load_joint_views():
        key = (pose, view)
        if key not in board_of:
            continue
        captures.append(_Capture(pose, view, theta, T_base_tcp,
                                 board_of[key], mad_of[key]))
    return captures


def _captures_from_cache(cache_p, board_of, mad_of, already_loaded):
    """Legacy fallback for captures made before joints.json was logged.

    Optional, and unnecessary for anything captured with this repo -- such
    views cannot carry joint offsets either, which is why the cache was
    retired.
    """
    if not cache_p or not Path(cache_p).exists():
        return []

    seen = set()
    for capture in already_loaded:
        seen.add((capture.pose, capture.view))

    cache = json.load(open(cache_p))
    captures = []
    for pose in cache:
        for view, cached in cache[pose].items():
            key = (int(pose), view)
            if key in seen or key not in board_of:
                continue
            if cached.get("T_base_tcp") is None:
                continue
            captures.append(_Capture(key[0], view, None,
                                     np.asarray(cached["T_base_tcp"]),
                                     board_of[key], mad_of[key]))
    return captures


def _apply_joint_offsets(captures, offsets):
    """Recompute FK with joint offsets where joint angles are available."""
    for capture in captures:
        if capture.theta is None:
            continue
        capture.T_base_tcp = K.fk(np.asarray(capture.theta) + offsets)


def _captures_on_mount(captures, mount):
    """The camera mount changed at pose_4; the two are not comparable."""
    on_mount = []
    for capture in captures:
        if (capture.pose in MOUNT_A) == (mount == "A"):
            on_mount.append(capture)
    return on_mount


def _fit_mount(captures, gate, fitted_boards):
    """Shared X for one mount; folder board poses land in `fitted_boards`."""
    T_base_tcp = [c.T_base_tcp for c in captures]
    T_cam_board = [c.T_cam_board for c in captures]
    folders = [c.pose for c in captures]

    # Downweight the views the per-view fit already distrusts, so a flipped
    # pose cannot pull the shared X it is being measured against.
    weights = []
    for capture in captures:
        trusted = (capture.mad_px or 0) <= gate
        weights.append(1.0 if trusted else 0.05)

    X, boards = fit_x_and_boards(T_base_tcp, T_cam_board, folders,
                                 w=np.array(weights))
    fitted_boards.update(boards)
    return X


def _keep_where(poses, keep_flags):
    kept = []
    for pose, keep in zip(poses, keep_flags):
        if keep:
            kept.append(pose)
    return kept


def analyse(groups, k=3.0, floor=1.0, verbose=True):
    """Per-folder scatter, before and after robust outlier rejection.

    The cut is median + k * MAD on the corner deviation from consensus, with a
    floor so a tight folder does not reject its own noise. MAD rather than std
    because std is what the outliers are corrupting -- using it to detect them
    is circular.
    """
    rep = {}
    for p in sorted(groups):
        items = groups[p]
        Ts = [t for _, t, _, _ in items]
        C = consensus(Ts)
        dev = np.array([corner_dev(T, C) for T in Ts])

        med = np.median(dev)
        mad = np.median(np.abs(dev - med))
        cut = max(med + k * 1.4826 * mad, floor)
        keep = dev <= cut

        # recompute consensus on the survivors, then re-measure
        if keep.sum() >= 3:
            C2 = consensus(_keep_where(Ts, keep))
            dev2 = np.array([corner_dev(T, C2) for T in Ts])
        else:
            C2, dev2, keep = C, dev, np.ones(len(dev), bool)

        t = np.array([T[:3, 3] for T in Ts]) * 1000
        std_all = float(np.sqrt(((t - t.mean(0)) ** 2).sum(1).mean()))
        tk = t[keep]
        std_keep = float(np.sqrt(((tk - tk.mean(0)) ** 2).sum(1).mean()))

        rej = []
        for i in np.nonzero(~keep)[0]:
            ang, az = yaw_about_normal(Ts[i], C2)
            rej.append({"view": items[i][0], "dev_mm": float(dev2[i]),
                        "rot_deg": float(ang), "axis_z": float(az),
                        "mad_px": items[i][2]})
        rep[p] = {"n": len(items), "n_rejected": int((~keep).sum()),
                  "cut_mm": float(cut),
                  "std_all_mm": std_all, "std_keep_mm": std_keep,
                  "dev_median_mm": float(np.median(dev2[keep])),
                  "dev_p90_mm": float(np.percentile(dev2[keep], 90)),
                  "rejected": rej,
                  "kept_views": [items[i][0] for i in np.nonzero(keep)[0]]}

        if verbose:
            print(f"  pose_{p:<3} n={len(items):3d}  "
                  f"std {std_all:6.2f} -> {std_keep:5.2f} mm   "
                  f"dev median {np.median(dev2[keep]):5.2f}  "
                  f"p90 {np.percentile(dev2[keep], 90):5.2f}   "
                  f"rejected {(~keep).sum()}")
            for r in rej:
                print(f"        {r['view']:16} dev {r['dev_mm']:8.1f} mm  "
                      f"rot {r['rot_deg']:6.2f} deg (axis_z {r['axis_z']:.2f})"
                      f"  fit mad {r['mad_px']:.2f} px")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "out" / "bundle_rims6_v2.json"))
    ap.add_argument("--cache", default=str(HERE / "out" / "cached_views.json"))
    ap.add_argument("--k", type=float, default=3.0,
                    help="reject at median + k*MAD of the corner deviation")
    ap.add_argument("--floor", type=float, default=1.0,
                    help="never reject below this deviation (mm)")
    ap.add_argument("--offsets", default=None,
                    help="comma-separated 6 joint offsets in deg, e.g. from "
                         "kinematic_calib.json")
    ap.add_argument("--gate", type=float, default=1.2,
                    help="mad_px above which a view is downweighted "
                         "when fitting the shared X")
    ap.add_argument("--out", default=str(HERE / "out" / "board_scatter.json"))
    args = ap.parse_args()

    off = None
    if args.offsets:
        off = np.radians([float(s) for s in args.offsets.split(",")])

    groups, _ = board_poses(args.src, args.cache, use_offsets=off,
                            gate=args.gate)
    print(f"board pose scatter per folder "
          f"({'with' if off is not None else 'no'} joint offsets)\n")
    rep = analyse(groups, k=args.k, floor=args.floor)

    a = np.array([rep[p]["std_all_mm"] for p in rep])
    b = np.array([rep[p]["std_keep_mm"] for p in rep])
    nrej = sum(rep[p]["n_rejected"] for p in rep)
    ntot = sum(rep[p]["n"] for p in rep)
    print(f"\n  overall: median folder std {np.median(a):.2f} -> "
          f"{np.median(b):.2f} mm   worst {a.max():.2f} -> {b.max():.2f} mm"
          f"   ({nrej}/{ntot} views rejected)")

    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
