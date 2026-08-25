#!/usr/bin/env python3
"""
kinematics.py - identify the arm's forward kinematics from the recorded data.

The kinematic error model needs FK as a *function of joint angles*, not just the
FK output that was logged. The URDF is not in this repository, so the model is
identified from the data.

The joint names are UR ones, so the arm is a UR and the only question is which.
That is settled without any optimisation: if the model is right then

    T_base_arm = T_base_tcp @ inv(FK(theta))

is the constant base mounting transform and must be identical for every
capture, so its scatter over the captures IS the model error. Measured over 171
views:

    UR5     0.24 mm      <- this arm
    UR5e    4.96 mm
    UR16e  15.77 mm
    UR10e  71.01 mm,  UR3 73.32,  UR3e 75.57,  UR10 75.95

and the recovered offset [-155.2, -195.9, +1400.3] mm reproduces the
`right_arm_base_link` transform hardcoded in normalize_frames.py to 0.3 mm --
an independent confirmation of both the arm model and the frame normalisation.

A general product-of-exponentials fit was tried first and is kept below, but it
is not used: with 42 free parameters and a hand-guessed seed it converged to a
41.7 mm residual, i.e. not at all. Identifying the arm instead of fitting a
generic chain is both exact and instant.

Joint order matters and is not the order in joints.json. That file lists
    elbow, shoulder_lift, shoulder_pan, wrist_1, wrist_2, wrist_3
while the kinematic chain is
    shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
so the columns are permuted by JOINT_ORDER below.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Joint names as they appear in joints.json, IN KINEMATIC ORDER. The prefix is
# whatever your robot's driver publishes -- ours is a dual-arm setup, hence
# "right_arm_". A bare UR driver publishes "shoulder_pan_joint" etc. Set
# JOINT_PREFIX to match; load_joint_views() skips any capture whose joints.json
# does not carry all six, so a wrong prefix shows up as "0 captures", not as
# silently wrong numbers.
JOINT_PREFIX = "right_arm_"
ARM_JOINTS = [JOINT_PREFIX + n for n in
              ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]]


# ---------------------------------------------------------------- SE(3)

def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    K = hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def exp_se3(xi):
    """xi = (w, v); returns the 4x4 homogeneous transform."""
    w, v = np.asarray(xi[:3], float), np.asarray(xi[3:], float)
    th = np.linalg.norm(w)
    T = np.eye(4)
    if th < 1e-12:
        T[:3, 3] = v
        return T
    K = hat(w / th)
    # p = V v with V = I + ((1-cos th)/th) K + ((th-sin th)/th) K^2
    V = np.eye(3) + (1 - np.cos(th)) / th * K + (th - np.sin(th)) / th * K @ K
    T[:3, :3] = exp_so3(w)
    T[:3, 3] = V @ v
    return T


def log_se3(T):
    """Inverse of exp_se3, returned as (w, v)."""
    R, p = T[:3, :3], T[:3, 3]
    c = (np.trace(R) - 1) / 2
    th = np.arccos(np.clip(c, -1, 1))
    if th < 1e-9:
        return np.concatenate([np.zeros(3), p])
    w_hat = (R - R.T) / (2 * np.sin(th)) * th
    w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])
    K = hat(w / th)
    Vinv = (np.eye(3) - 0.5 * K * th
            + (1 - th * np.cos(th / 2) / (2 * np.sin(th / 2))) * K @ K)
    return np.concatenate([w, Vinv @ p])


def screw(w, q):
    """Screw axis of a revolute joint: direction w through the point q."""
    w = w / max(np.linalg.norm(w), 1e-12)
    return np.concatenate([w, -np.cross(w, q)])


# ------------------------------------------------------------- the arm model

# Standard Denavit-Hartenberg (d, a, alpha) per joint; theta is the variable.
# The whole UR family is listed because identify_arm() picks between them, and
# because that identification is the only evidence that the model is right at
# all -- see the module docstring.
#
# ADDING A NON-UR ARM: append its (d, a, alpha) triple here under its own name
# and pass robot="<name>" (or set ACTIVE_ROBOT). Any 6R arm expressible in
# standard DH works unchanged. If yours is not -- a 7-DoF arm, or one with a
# non-DH-able chain -- use fit_poe() at the bottom of this file instead, which
# identifies a generic product-of-exponentials chain, and give it a decent seed.
DH = {
    #          d [m]                                   a [m]                        alpha [rad]
    "ur3":   ([0.1519, 0, 0, 0.11235, 0.08535, 0.0819], [0, -0.24365, -0.21325, 0, 0, 0], [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]),
    "ur5":   ([0.089159, 0, 0, 0.10915, 0.09465, 0.0823], [0, -0.425, -0.39225, 0, 0, 0], [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]),
    "ur10":  ([0.1273, 0, 0, 0.163941, 0.1157, 0.0922], [0, -0.612, -0.5723, 0, 0, 0], [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]),
    "ur3e":  ([0.15185, 0, 0, 0.13105, 0.08535, 0.0921], [0, -0.24355, -0.2132, 0, 0, 0], [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]),
    "ur5e":  ([0.1625, 0, 0, 0.1333, 0.0997, 0.0996], [0, -0.425, -0.3922, 0, 0, 0], [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]),
    "ur10e": ([0.1807, 0, 0, 0.17415, 0.11985, 0.11655], [0, -0.6127, -0.57155, 0, 0, 0], [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]),
    "ur16e": ([0.1807, 0, 0, 0.17415, 0.11985, 0.11655], [0, -0.4784, -0.36, 0, 0, 0], [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]),
}
DH = {k: tuple(np.asarray(x, float) for x in v) for k, v in DH.items()}

# The arm the reference dataset was shot on. Override per call with fk(...,
# robot=), or globally by assigning to this name before anything calls fk().
ACTIVE_ROBOT = "ur5"


def dh(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca,  st * sa, a * ct],
                     [st,  ct * ca, -ct * sa, a * st],
                     [0.0,      sa,       ca,      d],
                     [0.0,     0.0,      0.0,    1.0]])


def fk(theta, robot=None):
    """Tool pose in the ARM base frame, for a 6R arm in standard DH."""
    d, a, alpha = DH[robot or ACTIVE_ROBOT]
    T = np.eye(4)
    for i in range(6):
        T = T @ dh(theta[i], d[i], a[i], alpha[i])
    return T


def identify_arm(views, candidates=None):
    """Which arm is this? Rank the candidates by base-transform scatter.

    T_base_arm = T_base_tcp @ inv(fk(theta)) is the constant mounting transform,
    so if the model is right it is identical for every capture and its scatter
    over the captures IS the model error. No optimisation, no seed, no URDF.

    Returns [(name, median_scatter_mm), ...] best first. A correct model sits
    well under a millimetre; the next-best UR was 20x worse on our data.
    """
    out = []
    for name in (candidates or DH):
        _, sc = fit_base_transform(views, robot=name)
        out.append((name, float(np.median(sc))))
    return sorted(out, key=lambda kv: kv[1])


def fit_base_transform(views, theta_of=None, robot=None):
    """The constant T_base_arm implied by each capture, and its scatter."""
    theta_of = theta_of or (lambda v: v[2])
    Bs = [v[3] @ np.linalg.inv(fk(theta_of(v), robot)) for v in views]
    t = np.array([B[:3, 3] for B in Bs])
    R = exp_so3(np.mean([log_se3(B)[:3] for B in Bs], axis=0))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t.mean(0)
    scatter = np.linalg.norm(t - t.mean(0), axis=1) * 1000
    return T, scatter


# ---------------------------------------------------------------- data

def parse_pose_txt(path):
    """T_base_tcp from a capture's pose.txt (a dumped geometry_msgs Transform)."""
    c = Path(path).read_text()
    g = lambda k: float(re.search(rf"{k}:\s*([-\d.e+]+)", c).group(1))  # noqa: E731
    x, y, z, w = g("rotation_x"), g("rotation_y"), g("rotation_z"), g("rotation_w")
    T = np.eye(4)
    T[:3, :3] = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]])
    T[:3, 3] = [g("translation_x"), g("translation_y"), g("translation_z")]
    return T


def load_joint_views(poses=None, root=None):
    """(pose, view, theta[6], T_base_tcp, effort[6], base_frame) per capture."""
    root = root or (ROOT / "images" / "images_depth")
    out = []
    for pd in sorted(root.glob("pose_*"), key=lambda p: int(p.name.split("_")[1])):
        p = int(pd.name.split("_")[1])
        if poses is not None and p not in poses:
            continue
        for vd in sorted(pd.iterdir()):
            jf, pf = vd / "joints.json", vd / "pose.txt"
            if not (jf.exists() and pf.exists()):
                continue
            j = json.load(open(jf))
            idx = {n: i for i, n in enumerate(j["name"])}
            if not all(n in idx for n in ARM_JOINTS):
                continue
            th = np.array([j["position"][idx[n]] for n in ARM_JOINTS])
            eff = np.array(j["effort"][:6]) if len(j.get("effort", [])) >= 6 else np.zeros(6)
            out.append((p, vd.name, th, parse_pose_txt(pf), eff,
                        j.get("base_frame", "?")))
    return out


# ---------------------------------------------------------------- PoE fit

def poe_fk(par, th):
    """par = 6x6 joint params (w,q) flattened, then 6 for M as a se3 vector."""
    T = np.eye(4)
    for i in range(6):
        w, q = par[6 * i:6 * i + 3], par[6 * i + 3:6 * i + 6]
        T = T @ exp_se3(screw(w, q) * th[i])
    return T @ exp_se3(par[36:42])


def fit_poe(views, verbose=True):
    """Identify screw axes and home pose from (theta, T_base_tcp) pairs."""
    # seed: a generic 6R arm with alternating z/y axes, roughly UR-shaped
    seed = []
    axes = [[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, -1], [0, 1, 0]]
    pts = [[0, 0, 0], [0, 0, 0.15], [0.42, 0, 0.15],
           [0.81, 0, 0.15], [0.81, 0.11, 0.15], [0.81, 0.11, 0.06]]
    for a, q in zip(axes, pts):
        seed += list(a) + list(q)
    M0 = np.eye(4); M0[:3, 3] = [0.81, 0.19, 0.06]
    seed += list(log_se3(M0))
    x0 = np.array(seed, float)

    TH = np.array([v[2] for v in views])
    TT = np.array([v[3] for v in views])

    def resid(x):
        out = np.empty((len(views), 6))
        for k in range(len(views)):
            E = np.linalg.inv(poe_fk(x, TH[k])) @ TT[k]
            out[k] = log_se3(E)
        return out.ravel()

    sol = least_squares(resid, x0, method="lm", max_nfev=40000, xtol=1e-15,
                        ftol=1e-15)
    r = resid(sol.x).reshape(-1, 6)
    pos_mm = np.linalg.norm(r[:, 3:], axis=1) * 1000
    rot_deg = np.degrees(np.linalg.norm(r[:, :3], axis=1))
    if verbose:
        print(f"  PoE fit over {len(views)} views: "
              f"position residual median {np.median(pos_mm):.4f} mm "
              f"max {pos_mm.max():.4f} mm | rotation median "
              f"{np.median(rot_deg):.5f} deg max {rot_deg.max():.5f} deg")
    return sol.x, pos_mm, rot_deg


if __name__ == "__main__":
    views = load_joint_views()
    print(f"captures with joint data: {len(views)}")
    # NB: joints.json records the base_frame as captured and is NOT rewritten by
    # normalize_frames.py, so that field is stale. pose.txt is the normalised
    # source of truth; only pose_16 ("world") is still in a foreign frame.
    main_v = [v for v in views if v[0] != 16]
    print("\narm identification (base-transform scatter, lower is better):")
    for name, mm in identify_arm(main_v):
        print(f"    {name:<7} {mm:8.3f} mm" + ("   <- this arm" if mm == min(
            m for _, m in identify_arm(main_v)) else ""))
    T_arm, sc = fit_base_transform(main_v)
    print(f"\n{ACTIVE_ROBOT} + base transform over {len(main_v)} views "
          f"(all folders except pose_16):")
    print(f"  T_base_arm t = {np.round(T_arm[:3, 3] * 1000, 2)} mm")
    print(f"  scatter: median {np.median(sc):.3f}  p90 "
          f"{np.percentile(sc, 90):.3f}  max {sc.max():.3f} mm")
    for p_ in sorted({v[0] for v in main_v}):
        m = sc[[i for i, v in enumerate(main_v) if v[0] == p_]]
        print(f"    pose_{p_:<3} n={len(m):3d}  median {np.median(m):7.3f} mm")

    w = [v for v in views if v[0] == 16]
    if w:
        T16, sc16 = fit_base_transform(w)
        d = np.linalg.inv(T_arm) @ T16
        ang = np.degrees(np.linalg.norm(log_se3(d)[:3]))
        print(f"\npose_16 (\'world\' frame, {len(w)} views): own base transform "
              f"scatter median {np.median(sc16):.3f} mm")
        print(f"  world -> base_link: t {np.round(d[:3, 3] * 1000, 2)} mm, "
              f"rotation {ang:.4f} deg")
