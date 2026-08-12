#!/usr/bin/env python3
"""
kinematic_calib.py - joint-space error model for the arm.

The board pose is good to ~0.15 mm in the camera frame, but the same static
board scatters by 1-3 mm once pushed through T_base_tcp @ X. That residual is
configuration-dependent, not random, so it belongs to the robot, and the model
that can represent it has to live in joint space:

    theta_eff = theta + dtheta

    dtheta  constant joint offsets     (encoder zero errors)

A per-joint compliance term `+ C * tau`, driven by the reported effort, used to
be fitted alongside. It is GONE, and deliberately so:

  * it bought 4 um on top of the offsets (0.866 -> 0.862 mm median) for six
    extra parameters, and was never cross-validated;
  * fitted alone it returns C_4 = +0.80 deg/Nm. Joint-4 effort spans 0.65 units
    over all 184 views, so that term produces ~0.52 deg of "deflection" --
    essentially the fitted encoder offset dtheta_4 = -0.63 deg. It was
    impersonating the offset, not measuring stiffness. Turn the offsets on and
    C_4 collapses to -0.066;
  * the UR reports effort estimated from motor current, which includes friction
    and stiction and is not gravity-proportional anyway.

The decisive measurement: once the offsets are fitted, the per-view residual
correlates with NOTHING. Horizontal reach, |t_base_tcp|, TCP height, all six
joint angles, all six efforts, capture index within the folder and minutes into
the session all come in below |r| = 0.15 over 147 views. There is no
configuration-dependent signal left for any deflection model to remove; what
remains is vision tilt (see fit_board_multiview.py).

Identifiability, which sets what may be fitted at all:

  * dtheta_1 rotates the whole arm about the base z axis. That is a constant
    left-multiplication of the FK, absorbed exactly by the free per-folder
    T_base_board. NOT identifiable -- fixed at zero.
  * dtheta_6 rotates about the tool axis, absorbed exactly by the hand-eye X.
    NOT identifiable -- fixed at zero.
FK is recomputed from the joint angles with the identified UR5 model rather
than taken from the logged T_base_tcp -- that is the whole point, since the
logged value is a fixed number that no joint correction can act on. The arm's
base mounting transform is NOT needed: it is a constant left-multiplication,
absorbed exactly by the free per-folder board pose, which is also why pose_16
can be used despite still being logged in a 'world' frame.

X is deliberately SHARED across the folders being fitted rather than per-folder.
A per-folder X can absorb configuration-dependent error folder by folder, which
would hide exactly the effect being measured; sharing it forces the joint model
to explain the variation. This is only legitimate within one camera mount --
see the mount change between pose_3 and pose_4.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import kinematics as K  # noqa: E402
from handeye_board import CORNERS, se3, unse3, calibrate_hand_eye_park  # noqa: E402

# joints whose constant offset is identifiable (0-based: shoulder_lift..wrist_2)
FREE_OFFSET = [1, 2, 3, 4]


def flex_features(kind, TH, TB):
    """Regressors for a hand-eye that varies with arm configuration.

    TESTED AND REJECTED -- see test_wrist_flex.py. Kept because the machinery
    and, more importantly, the PLACEBO CONTROL are reusable for the next
    candidate. Nothing enables this by default. The verdict:

        model                LOFO median   folders improved
        joint offsets only      0.645 mm         --
        + wrist flex (th6)      0.515            11/11
        + gravity flex          0.406            11/11
        + PLACEBO fake_x        0.408            11/11   <-- no gravity in it
        + PLACEBO fake_y        0.452            11/11

    A direction with no gravitational meaning whatsoever scores identically to
    real gravity (0.408 vs 0.406). The reason is that in this dataset the camera
    always points at the board, so the arm's orientation lives on a narrow
    manifold and every "fixed direction expressed in the TCP frame" is nearly
    the same regressor: canonical correlations between the gravity and fake_x
    feature sets are 0.997 and 0.986. The data cannot tell gravity-driven sag
    apart from ANY smooth function of camera orientation.

    So leave-one-folder-out, on its own, is NOT a sufficient gate. It shows a
    parameter transfers between folders; it cannot show the parameter means what
    its name says. Every folder here uses similar arm configurations, so a
    view-dependent VISION error transfers just as well as a robot property
    would. Always run the placebo.

    What is real: ~0.24 mm of the residual is a reproducible function of camera
    orientation, common to every folder. It is not in the arm -- the fitted
    amplitude is 2 mm at the board on a 135 mm stick, and it grows without
    plateau as the amplitude prior is loosened (0.1 mm prior -> 0.624 LOFO,
    0.3 -> 0.543, 1.0 -> 0.406, 3.0 -> 0.352). It belongs in the vision model;
    see board_model.rim_bias() and the b1 term rejected there for the same
    pose-absorbing reason.

    The camera hangs on a stick in the gripper, so anything that changes how
    gravity loads that stick changes X. Two candidate regressors:

      wrist    [cos th6, sin th6] -- the tool roll, which swings the stick's
               mass about the tool axis. 12 parameters. Chosen because th6 is
               the strongest predictor of what is left in the residual
               (r = +0.337, against a max of 0.15 for everything else).
      gravity  the gravity direction in the TCP frame. 18 parameters. This is
               the old `flex` model from handeye_board.py, kept as the general
               comparator: if the wrist model wins it is because th6 is
               genuinely special, not just because gravity moved.
    """
    if kind == "wrist":
        return np.stack([np.cos(TH[:, 5]), np.sin(TH[:, 5])], 1)
    if kind in ("gravity", "fake_x", "fake_y"):
        # PLACEBO CONTROL. Real deflection is caused by GRAVITY, so it must be
        # a function of the gravity direction in the arm's frame and of nothing
        # else. fake_x and fake_y ask the same question about a direction that
        # has no gravitational meaning at all: they are pure viewing-geometry
        # regressors. If a fake direction predicts the residual as well as the
        # real one does, the model is not measuring sag -- it is absorbing
        # something that varies with how the camera is pointed, and the
        # gravitational story is a coincidence of this dataset.
        u = {"gravity": [0, 0, -1.0], "fake_x": [1.0, 0, 0],
             "fake_y": [0, 1.0, 0]}[kind]
        return np.array([T[:3, :3].T @ np.array(u) for T in TB])
    raise ValueError(kind)


def load(src, poses):
    """Match bundle board poses to joint captures, for the requested folders."""
    B = json.load(open(src))
    views = B["views"] if isinstance(B, dict) else B
    board = {(r["pose"], r["view"]): np.asarray(r["T_cam_board"]) for r in views}
    out = []
    for p, v, th, T, eff, bf in K.load_joint_views(poses=poses):
        if (p, v) in board:
            out.append((p, v, th, T, eff, board[(p, v)]))
    return out


def solve(views, use_offset=True, huber=0.003, fixed_dtheta=None,
          flex=None, fixed_flex=None, flex_prior=None):
    """Fit X, per-folder T_base_board and the joint error model together.

    flex: None, "wrist" or "gravity" -- see flex_features(). When set, the
    hand-eye becomes X_i = exp(M @ f_i) @ X, with M fitted (or held at
    fixed_flex, which is how leave-one-folder-out transfers it).

    flex_prior [m]: Gaussian prior on the deflection amplitude, as a
    displacement at the board. Unconstrained, this model is ill-conditioned --
    a large rotation about the TCP and a large translation nearly cancel at the
    board, so the fit happily runs to tens of degrees and hundreds of mm, which
    a 135 mm camera stick obviously cannot do. Constraining it to a physically
    possible amplitude is what makes the test meaningful rather than a
    demonstration that 12 free parameters can absorb 11 board poses.
    """
    poses = sorted({v[0] for v in views})
    pidx = {p: i for i, p in enumerate(poses)}
    TH = np.array([v[2] for v in views])
    TC = np.array([v[5] for v in views])
    vp = np.array([pidx[v[0]] for v in views])

    # seed from the MODEL FK so the seed lives in the same frame as the fit
    TB = np.array([K.ur5_fk(t) for t in TH])
    R_x, t_x = calibrate_hand_eye_park([T[:3, :3] for T in TB],
                                       [T[:3, 3] for T in TB],
                                       [T[:3, :3] for T in TC],
                                       [T[:3, 3] for T in TC])
    X0 = np.eye(4); X0[:3, :3] = R_x; X0[:3, 3] = t_x.ravel()

    seeds = []
    for p in poses:
        Ts = [TB[i] @ X0 @ TC[i] for i in range(len(views)) if vp[i] == pidx[p]]
        S = np.eye(4)
        U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
        S[:3, :3] = U @ Vt
        if np.linalg.det(S[:3, :3]) < 0:
            U[:, -1] *= -1; S[:3, :3] = U @ Vt
        S[:3, 3] = np.median([T[:3, 3] for T in Ts], axis=0)
        seeds.append(unse3(S))

    n_off = len(FREE_OFFSET) if use_offset else 0
    o_off = 6 + 6 * len(poses)
    o_flex = o_off + n_off
    F = flex_features(flex, TH, TB) if flex else None
    n_flex = 6 * F.shape[1] if (flex and fixed_flex is None) else 0
    x0 = np.concatenate([unse3(X0)] + seeds + [np.zeros(n_off + n_flex)])

    def theta_eff(x):
        th = TH.copy()
        if fixed_dtheta is not None:
            th = th + fixed_dtheta          # applied, not fitted
        if use_offset:
            d = np.zeros(6)
            d[FREE_OFFSET] = x[o_off:o_off + n_off]
            th = th + d
        return th

    def flex_M(x):
        if not flex:
            return None
        if fixed_flex is not None:
            return np.asarray(fixed_flex)
        return x[o_flex:o_flex + n_flex].reshape(6, F.shape[1])

    def resid(x):
        X = se3(x[0:3], x[3:6])
        Bs = [se3(x[6 + 6 * i:9 + 6 * i], x[9 + 6 * i:12 + 6 * i])
              for i in range(len(poses))]
        th = theta_eff(x)
        M = flex_M(x)
        out = np.empty((len(views), 12))
        for i in range(len(views)):
            Xi = X if M is None else se3(*np.split(M @ F[i], 2)) @ X
            A = K.ur5_fk(th[i]) @ Xi @ TC[i]
            B = Bs[vp[i]]
            out[i] = ((CORNERS @ A[:3, :3].T + A[:3, 3])
                      - (CORNERS @ B[:3, :3].T + B[:3, 3])).ravel()
        if flex and flex_prior and fixed_flex is None:
            lev = 0.47                      # m, camera to board
            pen = np.concatenate([M[:3].ravel() * lev, M[3:].ravel()])
            return np.concatenate([out.ravel(), pen / flex_prior * huber])
        return out.ravel()

    sol = least_squares(resid, x0, loss="soft_l1", f_scale=huber, max_nfev=300)
    nv12 = 12 * len(views)              # prior rows, if any, come after
    r = resid(sol.x)[:nv12].reshape(len(views), 4, 3)
    per = np.sqrt((r ** 2).sum(2).mean(1)) * 1000
    info = {"X": se3(sol.x[0:3], sol.x[3:6]),
            "per_view_mm": per,
            "poses": poses,
            "vp": vp,
            "dtheta": np.zeros(6),
            "flex": None}
    if use_offset:
        info["dtheta"][FREE_OFFSET] = sol.x[o_off:o_off + n_off]
    if flex:
        info["flex"] = (np.asarray(fixed_flex) if fixed_flex is not None
                        else sol.x[o_flex:o_flex + n_flex].reshape(6, F.shape[1]))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "out" / "bundle_rims6.json"))
    ap.add_argument("--poses", default="4,6,7,9,10,11,12,13,14,15,16",
                    help="folders to fit; default = camera mount B, excluding "
                         "pose_5 (degenerate) and pose_8 (broken)")
    ap.add_argument("--out", default=str(HERE / "out" / "kinematic_calib.json"))
    args = ap.parse_args()
    poses = [int(s) for s in args.poses.split(",")]

    print("checking the identified UR5 model against the logged FK ...")
    kv = [v for v in K.load_joint_views(poses=poses) if v[0] != 16]
    _, sc = K.fit_base_transform(kv)
    print(f"  UR5 reproduces the logged FK to median {np.median(sc):.3f} mm, "
          f"max {sc.max():.3f} mm over {len(kv)} views")
    if np.median(sc) > 1.0:
        print("  WARNING: the arm model does not match. Joint order or frame "
              "handling is wrong; the error model below would be meaningless.")

    views = load(args.src, poses)
    print(f"\n{len(views)} views with joints + a bundled board pose, "
          f"folders {sorted({v[0] for v in views})}")

    runs = [("baseline (no joint model)", False),
            ("joint offsets only", True)]
    # Reject gross outliers once, using the baseline model, and use the SAME
    # view set for every variant -- otherwise the comparison is between
    # different data as well as different models. The outliers are views whose
    # bundled board pose is simply wrong (max residual 465 mm on a board that
    # is 320 mm across); no joint model should be asked to absorb those.
    base0 = solve(views, use_offset=False)
    e0 = base0["per_view_mm"]
    keep = e0 < max(5 * np.median(e0), 5.0)
    if (~keep).sum():
        print(f"\nrejected {int((~keep).sum())} views with baseline residual "
              f"> {max(5 * np.median(e0), 5.0):.1f} mm "
              f"(worst {e0.max():.0f} mm): "
              + ", ".join(f"pose_{views[i][0]}/{views[i][1]}"
                          for i in np.nonzero(~keep)[0][:6])
              + (" ..." if (~keep).sum() > 6 else ""))
        views = [v for v, k in zip(views, keep) if k]

    best, out = None, {}
    for name, uo in runs:
        info = solve(views, use_offset=uo)
        e = info["per_view_mm"]
        print(f"\n### {name}")
        print(f"  board-corner residual: median {np.median(e):5.3f}  "
              f"p90 {np.percentile(e, 90):5.3f}  max {e.max():6.3f} mm")
        if uo:
            print("  joint offsets (deg): "
                  + "  ".join(f"j{i+1} {np.degrees(d):+.4f}"
                              for i, d in enumerate(info["dtheta"]) if d))
        t = info["X"][:3, 3] * 1000
        print(f"  X t [{t[0]:+8.2f} {t[1]:+8.2f} {t[2]:+8.2f}] mm  "
              f"|t| {np.linalg.norm(t):7.2f}")
        out[name] = {"median_mm": float(np.median(e)),
                     "p90_mm": float(np.percentile(e, 90)),
                     "X_tcp_cam": info["X"].tolist(),
                     "dtheta_deg": list(np.degrees(info["dtheta"]))}
        if best is None or np.median(e) < best[1]:
            best = (name, float(np.median(e)), info)

    # Leave-one-folder-out: fit the joint offsets on 10 folders, then apply them
    # UNCHANGED to the 11th, re-fitting only that folder's board pose and using
    # the training X. If the offsets are a real property of the arm they must
    # transfer; if they were absorbing per-folder error they will not.
    print("\n### leave-one-folder-out cross-validation (joint offsets)")
    fold_poses = sorted({v[0] for v in views})
    cv_base, cv_corr = [], []
    for p in fold_poses:
        tr = [v for v in views if v[0] != p]
        te = [v for v in views if v[0] == p]
        trained = solve(tr, use_offset=True)
        held_b = solve(te, use_offset=False)
        held_c = solve(te, use_offset=False, fixed_dtheta=trained["dtheta"])
        b, c = np.median(held_b["per_view_mm"]), np.median(held_c["per_view_mm"])
        cv_base.append(b); cv_corr.append(c)
        print(f"  held out pose_{p:<3} n={len(te):3d}   baseline {b:6.3f} -> "
              f"transferred offsets {c:6.3f} mm   {'better' if c < b else 'WORSE'}")
    print(f"  overall: baseline {np.median(cv_base):.3f} -> "
          f"{np.median(cv_corr):.3f} mm median across folds "
          f"({sum(1 for b, c in zip(cv_base, cv_corr) if c < b)}/"
          f"{len(cv_base)} folds improved)")
    out["cross_validation"] = {
        "baseline_median_mm": float(np.median(cv_base)),
        "transferred_median_mm": float(np.median(cv_corr)),
        "folds_improved": int(sum(1 for b, c in zip(cv_base, cv_corr) if c < b)),
        "folds": len(cv_base)}

    print(f"\nbest: {best[0]} at {best[1]:.3f} mm median")
    e = best[2]["per_view_mm"]
    print("per-folder residual (mm), best model:")
    for i, p in enumerate(best[2]["poses"]):
        m = e[best[2]["vp"] == i]
        print(f"  pose_{p:<3} n={len(m):3d}  median {np.median(m):6.3f}  "
              f"p90 {np.percentile(m, 90):6.3f}")
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
