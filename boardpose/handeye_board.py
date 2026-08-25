#!/usr/bin/env python3
"""
handeye_board.py - hand-eye calibration using the BOARD instead of the ArUco marker.

The camera is on a stick held by the gripper (eye-in-hand) and the board is
static within each pose_* folder, which is exactly the AX = XB setup:

    T_base_tcp(i) @ X @ T_cam_board(i) = T_base_board(p)      for all views i

    X            = T_tcp_cam, the hand-eye transform      (6 DoF, shared)
    T_base_board = the board pose in the robot base frame (6 DoF, per folder)

Why the board beats the marker here:

                        ArUco marker        board
    pose accuracy       ~1.0 mm / 0.38°     0.25 mm
    extent              ~100 mm             320 x 250 mm
    features            4 corners           40 contours, ~2400 points

The marker's ~1 mm was being injected straight into the hand-eye solution.

Residuals are reported as board-corner displacements in mm, not as an abstract
SE(3) norm, so they mean something. Because the stick flexes, the per-view
residual is also the measurement of how badly a RIGID hand-eye model fits --
that non-rigidity, not the calibration, is expected to dominate.

Validation is against the touch-probed corners in ground_truth.txt, which are
themselves only good to ~1.5-2.5 mm.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

V_TIP = np.array([-0.0043625, 0.00081629, 0.34239859])
CORNERS = np.array([[+0.160, -0.125, 0.025], [-0.160, -0.125, 0.025],
                    [-0.160, +0.125, 0.025], [+0.160, +0.125, 0.025]])
GT_ORDER = ["right bottom corner", "left bottom corner",
            "left top corner", "right top corner"]


def se3(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float))[0]
    T[:3, 3] = tvec
    return T


def unse3(T):
    return np.concatenate([cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]])


def calibrate_hand_eye_park(R_g, t_g, R_t, t_t):
    """Closed-form AX = XB (Park & Martin 1994), returning X = T_tcp_cam.

    Replaces cv2.calibrateHandEye, which is declared in the OpenCV 5 type stub
    but is not present in the compiled module -- so the cv2 call raised
    AttributeError and this file could not run at all after the OpenCV upgrade.
    Only a seed is needed here; solve() then refines X non-linearly against the
    board-corner residual, so the closed form does not set the final accuracy.

    Inputs match the cv2 signature: gripper-to-base and target-to-camera poses.
    """
    n = len(R_g)
    A, B = [], []
    for i in range(n - 1):
        for j in (i + 1,):
            Ta = np.eye(4); Ta[:3, :3] = R_g[j].T @ R_g[i]
            Ta[:3, 3] = R_g[j].T @ (t_g[i] - t_g[j])
            Tb = np.eye(4); Tb[:3, :3] = R_t[j] @ R_t[i].T
            Tb[:3, 3] = t_t[j] - Tb[:3, :3] @ t_t[i]
            if np.linalg.norm(cv2.Rodrigues(Ta[:3, :3])[0]) < 1e-3:
                continue          # a pure translation constrains no rotation
            A.append(Ta); B.append(Tb)
    if len(A) < 2:
        raise RuntimeError("hand-eye needs >=2 motions with real rotation")

    M = np.zeros((3, 3))
    for Ta, Tb in zip(A, B):
        al = cv2.Rodrigues(Ta[:3, :3])[0].ravel()
        be = cv2.Rodrigues(Tb[:3, :3])[0].ravel()
        M += np.outer(be, al)
    w, V = np.linalg.eigh(M.T @ M)
    R_x = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T @ M.T
    U, _, Vt = np.linalg.svd(R_x)          # re-orthonormalise
    R_x = U @ Vt
    if np.linalg.det(R_x) < 0:
        U[:, -1] *= -1; R_x = U @ Vt

    C, d = [], []
    for Ta, Tb in zip(A, B):
        C.append(Ta[:3, :3] - np.eye(3))
        d.append(R_x @ Tb[:3, 3] - Ta[:3, 3])
    t_x = np.linalg.lstsq(np.vstack(C), np.concatenate(d), rcond=None)[0]
    return R_x, t_x


def kabsch(P, Q):
    """Rigid transform mapping P onto Q."""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cq - R @ cp
    return T


def gt_board_poses():
    """Board pose in the base frame from the touch-probed corners."""
    out = {}
    for p in (1, 2, 3):
        txt = (ROOT / "images" / "images_depth" / f"pose_{p}" / "ground_truth.txt").read_text()
        blocks = re.split(r"(?m)^(\w[\w ]*corner):", txt)[1:]
        tips = {}
        for name, body in zip(blocks[0::2], blocks[1::2]):
            M = np.array(re.findall(
                r"(?m)^\s*(-?\d\.\d{3})\s+(-?\d\.\d{3})\s+(-?\d\.\d{3})\s+(-?\d\.\d{3})\s*$",
                body)[:3], float)
            tips[name.strip()] = M[:, 3] + M[:, :3] @ V_TIP
        P = np.array([tips[k] for k in GT_ORDER])
        out[p] = kabsch(CORNERS, P)
    return out


def load(src, cache, key):
    """Views with both robot FK and a camera-frame target pose."""
    views = []
    if key == "board":
        for r in json.load(open(src))["views"]:
            rec = cache[str(r["pose"])].get(r["view"])
            if rec and rec.get("T_base_tcp"):
                views.append((r["pose"], r["view"], np.asarray(rec["T_base_tcp"]),
                              np.asarray(r["T_cam_board"])))
    else:
        for p in cache:
            for v, rec in cache[p].items():
                if rec.get("T_base_tcp") and rec.get("T_cam_marker"):
                    views.append((int(p), v, np.asarray(rec["T_base_tcp"]),
                                  np.asarray(rec["T_cam_marker"])))
    return views


def solve(views, huber=0.003, reject=None, per_folder_x=False):
    """Solve AX = XB for the hand-eye X and the per-folder board poses.

    per_folder_x: give every folder its own hand-eye X_p instead of sharing one.
    The camera is on a stick in the gripper and can be disturbed or remounted
    between sessions, so a single X is only valid if the stick was never
    touched. Each folder is then an independent AX = XB problem (12 unknowns,
    13-25 views), which is identifiable provided the wrist actually rotated
    within the folder -- if it did not, X_p is under-determined and will drift
    to absorb the board pose instead. Compare the two fits before believing
    either: if X_p barely varies, the shared model is correct and cheaper.

    No deflection model is offered here any more. Two were tried and both
    failed -- see the DEFLECTION MODELS note at the bottom of this file. The
    configuration-dependent error they were aiming at is real, but it belongs
    to the joints, and kinematic_calib.py is where it is now fitted.

    Returns (X, B, per).
    """
    if reject:
        views = [v for v in views if (v[0], v[1]) not in reject]
    poses = sorted({v[0] for v in views})
    R_g = [v[2][:3, :3] for v in views]
    t_g = [v[2][:3, 3] for v in views]
    R_t = [v[3][:3, :3] for v in views]
    t_t = [v[3][:3, 3] for v in views]
    R_x, t_x = calibrate_hand_eye_park(R_g, t_g, R_t, t_t)
    X0 = np.eye(4); X0[:3, :3] = R_x; X0[:3, 3] = t_x.ravel()

    # Per-folder X must be SEEDED per folder. The closed form assumes one X for
    # everything it is given, so seeding 16 folders from one global solve puts
    # them all in the same wrong basin -- which is what happened: the mount was
    # changed between pose_3 and pose_4 (|t| 570 mm -> 133 mm), the global seed
    # landed on the majority value, and poses 1-3 were then rejected as
    # outliers despite being the correctly-solved ones.
    X0p = {}
    if per_folder_x:
        for p in poses:
            idx = [i for i, v in enumerate(views) if v[0] == p]
            Xp = np.eye(4)
            try:
                Rp_, tp_ = calibrate_hand_eye_park([R_g[i] for i in idx],
                                                   [t_g[i] for i in idx],
                                                   [R_t[i] for i in idx],
                                                   [t_t[i] for i in idx])
                Xp[:3, :3] = Rp_; Xp[:3, 3] = tp_.ravel()
            except Exception:
                Xp = X0.copy()          # too few motions in this folder
            X0p[p] = Xp

    # seed each folder's board pose from the median of the per-view predictions
    seeds = {}
    for p in poses:
        Xs = X0p.get(p, X0)
        Ts = [g @ Xs @ c for pp, _, g, c in views if pp == p]
        S = np.eye(4)
        U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
        S[:3, :3] = U @ Vt
        if np.linalg.det(S[:3, :3]) < 0:
            U[:, -1] *= -1; S[:3, :3] = U @ Vt
        S[:3, 3] = np.median([T[:3, 3] for T in Ts], axis=0)
        seeds[p] = S

    nb = 6 * len(poses)
    nx = 6 * len(poses) if per_folder_x else 6
    pidx = {p: i for i, p in enumerate(poses)}
    vpose = [pidx[v[0]] for v in views]
    # extra parameters, laid out after the hand-eye and 6*len(poses) board ones
    x0 = np.concatenate(([unse3(X0p[p]) for p in poses] if per_folder_x
                         else [unse3(X0)])
                        + [unse3(seeds[p]) for p in poses])

    def x_of(x, i):
        k = 6 * vpose[i] if per_folder_x else 0
        return se3(x[k:k + 3], x[k + 3:k + 6])

    def resid(x):
        B = {p: se3(x[nx + 6 * i:nx + 3 + 6 * i], x[nx + 3 + 6 * i:nx + 6 + 6 * i])
             for i, p in enumerate(poses)}
        out = []
        for i, (p, _, g, c) in enumerate(views):
            A = g @ x_of(x, i) @ c
            out.append(((CORNERS @ A[:3, :3].T + A[:3, 3])
                        - (CORNERS @ B[p][:3, :3].T + B[p][:3, 3])).ravel())
        return np.concatenate(out)

    sol = least_squares(resid, x0, loss="soft_l1", f_scale=huber, max_nfev=400)
    X = ({p: se3(sol.x[6 * i:6 * i + 3], sol.x[6 * i + 3:6 * i + 6])
          for i, p in enumerate(poses)} if per_folder_x
         else se3(sol.x[0:3], sol.x[3:6]))
    B = {p: se3(sol.x[nx + 6 * i:nx + 3 + 6 * i], sol.x[nx + 3 + 6 * i:nx + 6 + 6 * i])
         for i, p in enumerate(poses)}
    per = []
    for i, (p, v, g, c) in enumerate(views):
        A = g @ x_of(sol.x, i) @ c
        d = (CORNERS @ A[:3, :3].T + A[:3, 3]) - (CORNERS @ B[p][:3, :3].T + B[p][:3, 3])
        per.append({"pose": p, "view": v,
                    "reach_mm": float(np.linalg.norm(g[:3, 3]) * 1000),
                    "corner_rms_mm": float(np.sqrt((d ** 2).sum(1).mean()) * 1000),
                    "corner_max_mm": float(np.linalg.norm(d, axis=1).max() * 1000)})
    return X, B, per


# NOTE: a "base not level" rotation cannot be fitted here. It is a constant
# left-multiplication of the FK, so `tilt @ T_base_tcp @ X @ T_cam_board` can be
# matched exactly by rotating every T_base_board instead -- and those are free
# parameters. It is exactly unidentifiable against this objective. (Fitted
# anyway, it runs off to 55 deg and still lowers the residual.) It would only
# become observable against an external reference, i.e. the probed corners,
# which are good to 1.5-2.5 mm and cannot certify anything at this level.


# ---------------------------------------------------------- DEFLECTION MODELS
#
# Two gravity-deflection models used to live here. Both are DELETED, and the
# reason is worth more than the code was:
#
#   flex   let the hand-eye X vary linearly with the gravity direction in the
#          TCP frame. 18 parameters for 11%, and it moved X by 22 mm. It cannot
#          represent shoulder or elbow deflection at all -- that depends on the
#          whole arm configuration, not on the tool orientation.
#
#   droop  rotated the FK output about a horizontal axis through the shoulder by
#          an angle proportional to horizontal reach, i.e. the actual physics in
#          one parameter. Median AX=XB residual 2.19 -> 2.17 mm, the
#          residual-vs-reach correlation unchanged (+0.43 -> +0.44), and a
#          fitted rate of 1.18 deg/m (1.8 deg of sag at 1.5 m) that is
#          physically absurd. It was absorbing noise, not deflection.
#
# Why neither could work: these captures reach UPWARD, not outward.
# |t_base_tcp| is 1252-1522 mm but the horizontal reach is only 232-711 mm, so
# the gravity moment about the shoulder barely varies across the dataset. The
# "+0.68 correlation with reach" once quoted in the README is a correlation with
# |t|, dominated by TCP height rather than by lever arm.
#
# What replaced them: joint offsets, fitted in joint space from the joint angles
# capture_board_view.py now records (kinematic_calib.py). Those work --
# 1.34 -> 0.87 mm median, 11/11 leave-one-folder-out folds improved.
#
# And they finish the job. After the offsets, the per-view residual correlates
# with NOTHING: horizontal reach, |t|, TCP height, all six joint angles, all six
# efforts, capture order within the folder and minutes into the session all come
# in below |r| = 0.15 over 147 views. There is no deflection-shaped signal left.
# Do not reintroduce a deflection model here; measure first, and expect to find
# that what remains is vision tilt, not the arm (see fit_board_multiview.py).


def rot_mean(Rs):
    U, _, Vt = np.linalg.svd(sum(Rs))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    return R


def evaluate(name, X, board_views, gt):
    """Use hand-eye X with the BOARD vision measurements and score the result."""
    print(f"\n  evaluated with board vision, X from {name}:")
    tot = []
    for p in sorted({v[0] for v in board_views}):
        Xp = X.get(p) if isinstance(X, dict) else X
        if Xp is None:
            continue
        Ts = [g @ Xp @ c for pp, _, g, c in board_views if pp == p]
        M = np.eye(4)
        M[:3, :3] = rot_mean([T[:3, :3] for T in Ts])
        M[:3, 3] = np.median([T[:3, 3] for T in Ts], axis=0)
        # cross-view scatter of corner positions
        ref = CORNERS @ M[:3, :3].T + M[:3, 3]
        sc = [np.sqrt((((CORNERS @ T[:3, :3].T + T[:3, 3]) - ref) ** 2).sum(1).mean()) * 1000
              for T in Ts]
        
        if p in gt and gt[p] is not None:
            d = ref - (CORNERS @ gt[p][:3, :3].T + gt[p][:3, 3])
            dR = gt[p][:3, :3].T @ M[:3, :3]
            ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
            off = (M[:3, 3] - gt[p][:3, 3]) * 1000
            tot.append(np.median(sc))
            print(f"    pose_{p:<2d}: n={len(Ts):2d}  cross-view scatter med {np.median(sc):5.2f} "
                  f"p90 {np.percentile(sc, 90):6.2f} mm   |  vs probe: corner RMS "
                  f"{np.sqrt((d ** 2).sum(1).mean()) * 1000:6.2f} mm  "
                  f"dxyz [{off[0]:+6.1f} {off[1]:+6.1f} {off[2]:+6.1f}]  rot {ang:4.2f} deg")
        else:
            tot.append(np.median(sc))
            print(f"    pose_{p:<2d}: n={len(Ts):2d}  cross-view scatter med {np.median(sc):5.2f} "
                  f"p90 {np.percentile(sc, 90):6.2f} mm   |  vs probe: (no probe GT)")
    return float(np.mean(tot)) if tot else 0.0


def x_groups(X, tol=0.030):
    """Cluster per-folder hand-eye solutions by translation.

    Reporting one spread over all folders is misleading once the camera has
    actually been remounted: poses 1-3 sit at |t| = 570 mm and poses 4-16 at
    |t| = 134 mm, so a pooled standard deviation (554 mm in z) describes the
    remount, not the calibration. What matters is the spread WITHIN a mount,
    which is the repeatability of the hand-eye itself.

    Single-link clustering at `tol` metres of translation.
    """
    ps = sorted(X)
    T = {p: X[p][:3, 3] for p in ps}
    groups = []
    for p in ps:
        for g in groups:
            if any(np.linalg.norm(T[p] - T[q]) < tol for q in g):
                g.append(p); break
        else:
            groups.append([p])
    out = []
    for g in sorted(groups, key=lambda g: -len(g)):
        A = np.array([T[p] for p in g]) * 1000
        Rs = [X[p][:3, :3] for p in g]
        Rm = rot_mean(Rs)
        ang = max(np.linalg.norm(np.degrees(cv2.Rodrigues(Rm.T @ R)[0]))
                  for R in Rs) if len(g) > 1 else 0.0
        out.append((g, A.mean(0), A.std(0), np.ptp(A, axis=0), ang))
    return out


def report(name, X, per):
    e = np.array([r["corner_rms_mm"] for r in per])
    if isinstance(X, dict):
        print(f"\n### hand-eye from {name}   ({len(per)} views, "
              f"{len(X)} per-folder X)")
        print(f"  AX=XB residual (board corner): median {np.median(e):5.2f}  "
              f"p90 {np.percentile(e, 90):5.2f}  max {e.max():7.2f} mm")
        for g, mu, sd, rng, ang in x_groups(X):
            tag = "MOUNT" if len(g) > 1 else "isolated (suspect)"
            print(f"\n  {tag}: pose_{', pose_'.join(str(p) for p in g)}")
            print(f"    mean X t [{mu[0]:+8.2f} {mu[1]:+8.2f} {mu[2]:+8.2f}] mm"
                  f"   |t| {np.linalg.norm(mu):7.2f} mm")
            if len(g) > 1:
                print(f"    within-mount spread: sd [{sd[0]:5.2f} {sd[1]:5.2f} "
                      f"{sd[2]:5.2f}] mm  range [{rng[0]:6.2f} {rng[1]:6.2f} "
                      f"{rng[2]:6.2f}] mm  rot max {ang:.3f} deg")
        return
    t = X[:3, 3] * 1000
    rv = np.degrees(cv2.Rodrigues(X[:3, :3])[0].ravel())
    print(f"\n### hand-eye from {name}   ({len(per)} views)")
    print(f"  X = T_tcp_cam   translation [{t[0]:+8.2f} {t[1]:+8.2f} {t[2]:+8.2f}] mm"
          f"   |t| {np.linalg.norm(t):7.2f} mm")
    print(f"                  rotation    [{rv[0]:+7.2f} {rv[1]:+7.2f} {rv[2]:+7.2f}] deg")
    print(f"  AX=XB residual (board corner): median {np.median(e):5.2f}  "
          f"p90 {np.percentile(e, 90):5.2f}  max {e.max():7.2f} mm")
    # a residual that still tracks reach means the deflection model has not
    reach = np.array([r.get("reach_mm", np.nan) for r in per])
    if np.isfinite(reach).all() and len(e) > 3:
        print(f"  residual vs reach: corr {np.corrcoef(reach, e)[0, 1]:+.2f}  "
              f"(reach {reach.min():.0f}-{reach.max():.0f} mm)")
    bad = [r for r in per if r["corner_rms_mm"] > 5 * np.median(e)]
    if bad:
        print(f"  outliers (>5x median): " +
              ", ".join(f"pose_{r['pose']}/{r['view']} {r['corner_rms_mm']:.0f}mm" for r in bad[:4]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "out" / "bundle_2level.json"))
    ap.add_argument("--out", default=str(HERE / "out" / "handeye.json"))
    ap.add_argument("--per-folder-x", action="store_true",
                    help="give each pose folder its own hand-eye X (the camera "
                         "stick may be disturbed between sessions)")
    args = ap.parse_args()

    cache = json.load(open(HERE / "out" / "cached_views.json"))
    gt = gt_board_poses()
    print("touch-probed board poses in the base frame (reference, itself ~1.5-2.5 mm):")
    for p in sorted(gt):
        print(f"  pose_{p}: origin [{gt[p][0, 3] * 1000:+8.2f} {gt[p][1, 3] * 1000:+8.2f} "
              f"{gt[p][2, 3] * 1000:+8.2f}] mm")

    board_views = load(args.src, cache, "board")
    results = {}
    for name, key, src in [("ArUco marker (old)", "marker", None),
                           ("BOARD (new)", "board", args.src)]:
        views = load(src, cache, key)
        if len(views) < 6:
            print(f"\n### {name}: only {len(views)} usable views, skipping")
            continue
        X, B, per = solve(views)
        report(name, X, per)
        bad = {(r["pose"], r["view"]) for r in per
               if r["corner_rms_mm"] > 5 * np.median([q["corner_rms_mm"] for q in per])}
        if bad:
            X, B, per = solve(views, reject=bad)
            report(name + ", outliers removed", X, per)
        evaluate(name, X, board_views, gt)
        results[name] = {"X_tcp_cam": X.tolist(), "per_view": per,
                         "T_base_target": {str(p): B[p].tolist() for p in B}}

        if key != "board":
            continue

        if args.per_folder_x:
            # NOT `reject=bad`: those outliers were judged against the shared X,
            # which is meaningless once the mount has changed between sessions.
            # Scored against one X, the folders with the minority mount look
            # like outliers and get discarded -- here that silently removed
            # poses 1-3, the very folders whose X was correct. Re-derive the
            # outliers from this model instead.
            Xp, Bp, perp = solve(views, per_folder_x=True)
            m = np.median([q["corner_rms_mm"] for q in perp])
            badp = {(r["pose"], r["view"]) for r in perp
                    if r["corner_rms_mm"] > 5 * m}
            if badp:
                Xp, Bp, perp = solve(views, reject=badp, per_folder_x=True)
            report(name + ", per-folder X", Xp, perp)
            evaluate(name + " per-folder X", Xp, board_views, gt)
            results["per_folder_x"] = {
                "X_tcp_cam": {str(p): Xp[p].tolist() for p in Xp},
                "per_view": perp,
                "T_base_target": {str(p): Bp[p].tolist() for p in Bp}}

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
