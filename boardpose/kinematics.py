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

# Below this rotation angle the small-angle branch is taken: the series
# denominators carry a sin(angle) that is not safely invertible.
_MIN_ROTATION_EXP = 1e-12
_MIN_ROTATION_LOG = 1e-9


def hat(omega):
    """Skew-symmetric matrix with hat(omega) @ v == cross(omega, v)."""
    return np.array([[0, -omega[2], omega[1]],
                     [omega[2], 0, -omega[0]],
                     [-omega[1], omega[0], 0]])


def exp_so3(omega):
    """Rotation vector -> rotation matrix (Rodrigues)."""
    angle = np.linalg.norm(omega)
    if angle < _MIN_ROTATION_EXP:
        return np.eye(3)
    skew = hat(omega / angle)
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


def exp_se3(xi):
    """xi = (omega, v); returns the 4x4 homogeneous transform."""
    omega = np.asarray(xi[:3], float)
    v = np.asarray(xi[3:], float)
    angle = np.linalg.norm(omega)

    transform = np.eye(4)
    if angle < _MIN_ROTATION_EXP:
        transform[:3, 3] = v
        return transform

    transform[:3, :3] = exp_so3(omega)
    transform[:3, 3] = _left_jacobian(omega / angle, angle) @ v
    return transform


def _left_jacobian(axis, angle):
    """V in p = V v, mapping the se(3) translation part onto the real one."""
    skew = hat(axis)
    return (np.eye(3)
            + (1 - np.cos(angle)) / angle * skew
            + (angle - np.sin(angle)) / angle * skew @ skew)


def _inverse_left_jacobian(axis, angle):
    skew = hat(axis)
    half = angle / 2
    return (np.eye(3)
            - 0.5 * skew * angle
            + (1 - angle * np.cos(half) / (2 * np.sin(half))) * skew @ skew)


def log_se3(T):
    """Inverse of exp_se3, returned as (omega, v)."""
    R, translation = T[:3, :3], T[:3, 3]
    cos_angle = (np.trace(R) - 1) / 2
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    if angle < _MIN_ROTATION_LOG:
        return np.concatenate([np.zeros(3), translation])

    omega_hat = (R - R.T) / (2 * np.sin(angle)) * angle
    omega = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])
    v = _inverse_left_jacobian(omega / angle, angle) @ translation
    return np.concatenate([omega, v])


def screw(axis, point_on_axis):
    """Screw axis of a revolute joint: direction `axis` through a point."""
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    return np.concatenate([axis, -np.cross(axis, point_on_axis)])


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
    """One standard Denavit-Hartenberg link transform."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    return np.array([[cos_t, -sin_t * cos_a,  sin_t * sin_a, a * cos_t],
                     [sin_t,  cos_t * cos_a, -cos_t * sin_a, a * sin_t],
                     [0.0,            sin_a,          cos_a,         d],
                     [0.0,              0.0,            0.0,       1.0]])


def fk(theta, robot=None):
    """Tool pose in the ARM base frame, for a 6R arm in standard DH."""
    link_d, link_a, link_alpha = DH[robot or ACTIVE_ROBOT]
    tool_pose = np.eye(4)
    for joint in range(6):
        tool_pose = tool_pose @ dh(theta[joint], link_d[joint],
                                   link_a[joint], link_alpha[joint])
    return tool_pose


def identify_arm(views, candidates=None):
    """Rank candidate arms by base-transform scatter, best first.

    T_base_arm = T_base_tcp @ inv(fk(theta)) is constant iff the model is
    right, so its scatter IS the model error -- no optimisation, seed or URDF.
    A correct model sits well under a millimetre.
    """
    ranking = []
    for name in (candidates or DH):
        _, scatter_mm = fit_base_transform(views, robot=name)
        ranking.append((name, float(np.median(scatter_mm))))
    return sorted(ranking, key=lambda name_and_scatter: name_and_scatter[1])


def fit_base_transform(views, theta_of=None, robot=None):
    """The constant T_base_arm implied by each capture, and its scatter."""
    theta_of = theta_of or (lambda view: view[2])
    per_view_base = [view[3] @ np.linalg.inv(fk(theta_of(view), robot))
                     for view in views]

    translations = np.array([T[:3, 3] for T in per_view_base])
    mean_translation = translations.mean(0)
    mean_rotation = exp_so3(
        np.mean([log_se3(T)[:3] for T in per_view_base], axis=0))

    T_base_arm = np.eye(4)
    T_base_arm[:3, :3] = mean_rotation
    T_base_arm[:3, 3] = mean_translation
    scatter_mm = np.linalg.norm(translations - mean_translation, axis=1) * 1000
    return T_base_arm, scatter_mm


# ---------------------------------------------------------------- data

def parse_pose_txt(path):
    """T_base_tcp from a capture's pose.txt (a dumped geometry_msgs Transform)."""
    text = Path(path).read_text()

    def field(name):
        return float(re.search(rf"{name}:\s*([-\d.e+]+)", text).group(1))

    x, y, z, w = (field("rotation_x"), field("rotation_y"),
                  field("rotation_z"), field("rotation_w"))
    T_base_tcp = np.eye(4)
    T_base_tcp[:3, :3] = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]])
    T_base_tcp[:3, 3] = [field("translation_x"), field("translation_y"),
                         field("translation_z")]
    return T_base_tcp


def load_joint_views(poses=None, root=None):
    """(pose, view, theta[6], T_base_tcp, effort[6], base_frame) per capture."""
    root = root or (ROOT / "images" / "images_depth")
    views = []
    for pose_dir in sorted(root.glob("pose_*"), key=_pose_number):
        pose = _pose_number(pose_dir)
        if poses is not None and pose not in poses:
            continue
        for view_dir in sorted(pose_dir.iterdir()):
            view = _read_joint_view(pose, view_dir)
            if view is not None:
                views.append(view)
    return views


def _pose_number(pose_dir):
    return int(pose_dir.name.split("_")[1])


def views_in_pose(views, pose):
    """Captures belonging to one pose_N folder."""
    selected = []
    for view in views:
        if view[0] == pose:
            selected.append(view)
    return selected


def split_out_pose(views, pose):
    """(captures in pose_N, everything else)."""
    inside, outside = [], []
    for view in views:
        if view[0] == pose:
            inside.append(view)
        else:
            outside.append(view)
    return inside, outside


def _read_joint_view(pose, view_dir):
    """One capture, or None if it lacks joints, a pose, or any of ARM_JOINTS."""
    joints_path, pose_path = view_dir / "joints.json", view_dir / "pose.txt"
    if not (joints_path.exists() and pose_path.exists()):
        return None

    joints = json.load(open(joints_path))
    column_of = {name: i for i, name in enumerate(joints["name"])}
    if not all(name in column_of for name in ARM_JOINTS):
        return None

    # Reordered into kinematic order; joints.json is alphabetical.
    theta = np.array([joints["position"][column_of[name]] for name in ARM_JOINTS])
    effort = joints.get("effort", [])
    effort = np.array(effort[:6]) if len(effort) >= 6 else np.zeros(6)
    return (pose, view_dir.name, theta, parse_pose_txt(pose_path), effort,
            joints.get("base_frame", "?"))


# ---------------------------------------------------------------- PoE fit

def poe_fk(par, th):
    """par = 6x6 joint params (w,q) flattened, then 6 for M as a se3 vector."""
    tool_pose = np.eye(4)
    for joint in range(6):
        axis = par[6 * joint:6 * joint + 3]
        point_on_axis = par[6 * joint + 3:6 * joint + 6]
        tool_pose = tool_pose @ exp_se3(screw(axis, point_on_axis) * th[joint])
    home_pose = exp_se3(par[36:42])
    return tool_pose @ home_pose


def fit_poe(views, verbose=True):
    """Identify screw axes and home pose from (theta, T_base_tcp) pairs."""
    seed = _generic_6r_seed()
    all_theta = np.array([view[2] for view in views])
    all_T_base_tcp = np.array([view[3] for view in views])

    def residual(params):
        errors = np.empty((len(views), 6))
        for k in range(len(views)):
            predicted_to_measured = (np.linalg.inv(poe_fk(params, all_theta[k]))
                                     @ all_T_base_tcp[k])
            errors[k] = log_se3(predicted_to_measured)
        return errors.ravel()

    solution = least_squares(residual, seed, method="lm", max_nfev=40000,
                             xtol=1e-15, ftol=1e-15)
    per_view = residual(solution.x).reshape(-1, 6)
    pos_mm = np.linalg.norm(per_view[:, 3:], axis=1) * 1000
    rot_deg = np.degrees(np.linalg.norm(per_view[:, :3], axis=1))
    if verbose:
        print(f"  PoE fit over {len(views)} views: "
              f"position residual median {np.median(pos_mm):.4f} mm "
              f"max {pos_mm.max():.4f} mm | rotation median "
              f"{np.median(rot_deg):.5f} deg max {rot_deg.max():.5f} deg")
    return solution.x, pos_mm, rot_deg


def _generic_6r_seed():
    """A roughly UR-shaped 6R arm: alternating z/y axes, plausible link points."""
    axes = [[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, -1], [0, 1, 0]]
    points_on_axes = [[0, 0, 0], [0, 0, 0.15], [0.42, 0, 0.15],
                      [0.81, 0, 0.15], [0.81, 0.11, 0.15], [0.81, 0.11, 0.06]]
    seed = []
    for axis, point in zip(axes, points_on_axes):
        seed += list(axis) + list(point)

    home_pose = np.eye(4)
    home_pose[:3, 3] = [0.81, 0.19, 0.06]
    seed += list(log_se3(home_pose))
    return np.array(seed, float)


def _row_indices_for_pose(views, pose):
    """Positions in `views` belonging to one folder, for indexing arrays."""
    rows = []
    for i, view in enumerate(views):
        if view[0] == pose:
            rows.append(i)
    return rows


def _report_arm_identification(views):
    ranking = identify_arm(views)
    best_scatter_mm = ranking[0][1]
    print("\narm identification (base-transform scatter, lower is better):")
    for name, scatter_mm in ranking:
        marker = "   <- this arm" if scatter_mm == best_scatter_mm else ""
        print(f"    {name:<7} {scatter_mm:8.3f} mm{marker}")


def _report_base_transform(views):
    T_base_arm, scatter_mm = fit_base_transform(views)
    print(f"\n{ACTIVE_ROBOT} + base transform over {len(views)} views "
          f"(all folders except pose_16):")
    print(f"  T_base_arm t = {np.round(T_base_arm[:3, 3] * 1000, 2)} mm")
    print(f"  scatter: median {np.median(scatter_mm):.3f}  p90 "
          f"{np.percentile(scatter_mm, 90):.3f}  max {scatter_mm.max():.3f} mm")
    for pose in sorted({view[0] for view in views}):
        rows = _row_indices_for_pose(views, pose)
        print(f"    pose_{pose:<3} n={len(rows):3d}  "
              f"median {np.median(scatter_mm[rows]):7.3f} mm")
    return T_base_arm


def _report_foreign_frame(views, T_base_arm):
    """pose_16 is still in the 'world' frame; recover its offset as a check."""
    if not views:
        return
    T_world_arm, scatter_mm = fit_base_transform(views)
    world_to_base = np.linalg.inv(T_base_arm) @ T_world_arm
    rotation_deg = np.degrees(np.linalg.norm(log_se3(world_to_base)[:3]))
    print(f"\npose_16 ('world' frame, {len(views)} views): own base transform "
          f"scatter median {np.median(scatter_mm):.3f} mm")
    print(f"  world -> base_link: t "
          f"{np.round(world_to_base[:3, 3] * 1000, 2)} mm, "
          f"rotation {rotation_deg:.4f} deg")


if __name__ == "__main__":
    all_views = load_joint_views()
    print(f"captures with joint data: {len(all_views)}")

    # NB: joints.json records the base_frame as captured and is NOT rewritten by
    # normalize_frames.py, so that field is stale. pose.txt is the normalised
    # source of truth; only pose_16 ("world") is still in a foreign frame.
    world_views, base_link_views = split_out_pose(all_views, 16)

    _report_arm_identification(base_link_views)
    _report_foreign_frame(world_views, _report_base_transform(base_link_views))
