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
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import kinematics as K  # noqa: E402
from handeye_board import (CORNERS, se3, unse3,  # noqa: E402
                           calibrate_hand_eye_park)
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
    for p in folders:
        Ts = [Tb[i] @ X0 @ Tc[i] for i in range(len(Tb)) if gi[i] == fidx[p]]
        seeds.append(unse3(consensus(Ts)))
    x0 = np.concatenate([unse3(X0)] + seeds)

    A = np.array([Tb[i] @ np.eye(4) for i in range(len(Tb))])

    def resid(x):
        X = se3(x[0:3], x[3:6])
        Bs = [se3(x[6 + 6 * i:9 + 6 * i], x[9 + 6 * i:12 + 6 * i])
              for i in range(len(folders))]
        out = np.empty((len(Tb), 12))
        for i in range(len(Tb)):
            P = A[i] @ X @ Tc[i]
            B = Bs[gi[i]]
            out[i] = w[i] * ((CORNERS @ P[:3, :3].T + P[:3, 3])
                             - (CORNERS @ B[:3, :3].T + B[:3, 3])).ravel()
        return out.ravel()

    sol = least_squares(resid, x0, loss="soft_l1", f_scale=huber, max_nfev=300)
    X = se3(sol.x[0:3], sol.x[3:6])
    boards = {p: se3(sol.x[6 + 6 * i:9 + 6 * i], sol.x[9 + 6 * i:12 + 6 * i])
              for p, i in fidx.items()}
    return X, boards


def consensus(Ts):
    """Robust average pose of a set: chordal-mean rotation, median translation."""
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    C = np.eye(4)
    C[:3, :3], C[:3, 3] = R, np.median([T[:3, 3] for T in Ts], axis=0)
    return C


def corner_dev(T, C):
    """Deviation of pose T from consensus C, as an RMS over the board corners.

    Corners rather than the origin: an origin-only distance is blind to a pure
    rotation about the origin, which is exactly what a yaw flip is.
    """
    a = CORNERS @ T[:3, :3].T + T[:3, 3]
    b = CORNERS @ C[:3, :3].T + C[:3, 3]
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()) * 1000)


def yaw_about_normal(T, C):
    """Rotation of T relative to C, decomposed into (about board z, tilt) deg."""
    D = C[:3, :3].T @ T[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1)))
    # axis component along board z tells a yaw flip from a tilt error
    w = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
    n = np.linalg.norm(w)
    axis_z = abs(w[2] / n) if n > 1e-9 else 1.0
    return ang, axis_z


def board_poses(src, cache_p=None, use_offsets=None, gate=1.2):
    """T_base_board per view, grouped by folder, with a per-mount hand-eye X."""
    B = json.load(open(src))
    views = B["views"] if isinstance(B, dict) else B
    board = {(r["pose"], r["view"]): np.asarray(r["T_cam_board"]) for r in views}
    mad = {(r["pose"], r["view"]): r.get("mad_px") for r in views}

    # T_base_tcp comes from the joint angles, so joint offsets can be applied.
    rows = []
    for p, v, th, T, eff, bf in K.load_joint_views():
        if (p, v) in board:
            rows.append([p, v, th, T, board[(p, v)], mad[(p, v)]])

    # Legacy fallback: captures made before capture_board_view.py logged
    # joints.json have only the FK output, cached externally. Optional, and
    # unnecessary for anything captured with this repo -- such views cannot
    # carry joint offsets either, which is why the cache was retired.
    if cache_p and Path(cache_p).exists():
        have = {(r[0], r[1]) for r in rows}
        cache = json.load(open(cache_p))
        for p in cache:
            for v, rec in cache[p].items():
                if (int(p), v) in have or (int(p), v) not in board:
                    continue
                if rec.get("T_base_tcp") is None:
                    continue
                rows.append([int(p), v, None, np.asarray(rec["T_base_tcp"]),
                             board[(int(p), v)], mad[(int(p), v)]])

    # FK: recomputed with joint offsets when asked and available, else logged.
    for r in rows:
        if use_offsets is not None and r[2] is not None:
            r[3] = K.fk(np.asarray(r[2]) + use_offsets)
        r.append(None)

    out, fitted = {}, {}
    for mount in ("A", "B"):
        sel = [r for r in rows if (r[0] in MOUNT_A) == (mount == "A")]
        if not sel:
            continue
        Tb, Tc = [r[3] for r in sel], [r[4] for r in sel]
        grp = [r[0] for r in sel]
        # Downweight the views the per-view fit already distrusts, so a flipped
        # pose cannot pull the shared X it is being measured against.
        w = np.array([1.0 if (r[5] or 0) <= gate else 0.05 for r in sel])
        X, boards = fit_x_and_boards(Tb, Tc, grp, w=w)
        fitted.update(boards)
        for r in sel:
            out.setdefault(r[0], []).append(
                (r[1], r[3] @ X @ r[4], r[5], r[4]))
    return out, fitted


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
            C2 = consensus([T for T, kk in zip(Ts, keep) if kk])
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
