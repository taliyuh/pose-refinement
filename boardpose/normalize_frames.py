#!/usr/bin/env python3
"""
normalize_frames.py - put every captured TCP pose into ONE base frame.

Why this exists: capture_board_view.py originally resolved its base frame by
trying PLANNING_FRAMES in order and keeping whichever answered first. Which one
answered first depended on TF timing at startup, so it varied per session. The
captures ended up split across three frames:

    base_footprint       pose_4, 5, 9, 10, 11
    base_link            pose_6, 8
    right_arm_base_link  pose_7

base_link -> base_footprint is [0, 0, -10 mm] with no rotation, so those two are
nearly interchangeable. base_link -> right_arm_base_link is [-155, -196, +1400] mm
with a 135 deg rotation about X, which is not.

Hand-eye itself survives this: T_base_board is free per folder, and a constant
left-multiplication of the FK within a folder is exactly absorbed by it -- the
same argument that makes a base tilt unidentifiable in handeye_board.py. What
does NOT survive is anything reading t_base_tcp directly: the droop model's
horizontal reach and azimuth, and the coverage gates. Those were computed across
mixed frames and are meaningless until this is run.

The correction is exact, not a fit. These are static transforms straight from
the URDF, so no accuracy is lost.

    T_target_tcp = T_target_source @ T_source_tcp

Dry run by default; --apply rewrites. Originals are preserved as pose_raw.txt and
the applied transform is recorded in meta.json, so it is reversible and a second
run is a no-op.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Static transforms, read from TF on the live robot with
#   ros2 run tf2_ros tf2_echo base_link <frame>
# Keyed by source frame, giving T_base_link_source.
T_BASE_FROM = {
    "base_link": (np.array([0.0, 0.0, 0.0]),
                  np.array([0.0, 0.0, 0.0, 1.0])),
    "base_footprint": (np.array([0.0, 0.0, -0.010]),
                       np.array([0.0, 0.0, 0.0, 1.0])),
    "right_arm_base_link": (np.array([-0.155, -0.196, 1.400]),
                            np.array([0.924, 0.0, 0.0, 0.383])),
}

POSE_TXT = """# Translation (x, y, z)
translation_x: {t[0]:.15f}
translation_y: {t[1]:.15f}
translation_z: {t[2]:.15f}

# Rotation Quaternion (x, y, z, w)
rotation_x: {q[0]:.15f}
rotation_y: {q[1]:.15f}
rotation_z: {q[2]:.15f}
rotation_w: {q[3]:.15f}

# Timestamp
secs: {secs}
nsecs: {nsecs}
"""


def quat_to_R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def to_T(t, q):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(np.asarray(q, float))
    T[:3, 3] = np.asarray(t, float)
    return T


def parse_pose(path):
    d = {}
    for ln in path.read_text().splitlines():
        if ":" in ln and not ln.strip().startswith("#"):
            k, v = ln.split(":", 1)
            d[k.strip()] = float(v)
    t = np.array([d["translation_x"], d["translation_y"], d["translation_z"]])
    q = np.array([d["rotation_x"], d["rotation_y"], d["rotation_z"], d["rotation_w"]])
    return t, q, int(d["secs"]), int(d["nsecs"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(HERE.parent / "images" / "images_depth"))
    ap.add_argument("--target", default="base_link")
    ap.add_argument("--apply", action="store_true", help="rewrite; otherwise dry run")
    args = ap.parse_args()

    if args.target not in T_BASE_FROM:
        ap.error(f"no static transform recorded for target {args.target!r}")
    T_bt = to_T(*T_BASE_FROM[args.target])
    T_target_base = np.linalg.inv(T_bt)          # target <- base_link

    root = Path(args.root)
    todo, done, skipped = [], 0, 0
    for pd in sorted(root.glob("pose_*"), key=lambda p: int(p.name.split("_")[1])):
        for v in sorted(pd.iterdir()):
            mf, pf = v / "meta.json", v / "pose.txt"
            if not (mf.is_file() and pf.is_file()):
                skipped += 1
                continue
            meta = json.loads(mf.read_text())
            src = meta.get("base_frame")
            if meta.get("frame_normalized_to") == args.target:
                done += 1
                continue
            if src == args.target:
                done += 1
                continue
            if src not in T_BASE_FROM:
                print(f"  NO TRANSFORM for {src!r} ({v}) -- add it to T_BASE_FROM")
                skipped += 1
                continue
            todo.append((v, meta, src))

    by_src = {}
    for v, meta, src in todo:
        by_src.setdefault(src, []).append((v, meta))
    print(f"root {root}")
    print(f"  already in {args.target}: {done}   to convert: {len(todo)}   skipped: {skipped}")
    for src, items in by_src.items():
        T_target_src = T_target_base @ to_T(*T_BASE_FROM[src])
        d = T_target_src[:3, 3] * 1000
        ang = np.degrees(np.arccos(np.clip((np.trace(T_target_src[:3, :3]) - 1) / 2, -1, 1)))
        print(f"  {src:22s} -> {args.target}: {len(items):3d} views, "
              f"shift [{d[0]:+.0f} {d[1]:+.0f} {d[2]:+.0f}] mm, rotation {ang:.1f} deg")
        sample = items[0][0]
        t, q, _, _ = parse_pose(sample / "pose.txt")
        Tn = T_target_src @ to_T(t, q)
        print(f"      e.g. {sample.parent.name}/{sample.name}: "
              f"t [{t[0]*1000:+.0f} {t[1]*1000:+.0f} {t[2]*1000:+.0f}] -> "
              f"[{Tn[0,3]*1000:+.0f} {Tn[1,3]*1000:+.0f} {Tn[2,3]*1000:+.0f}] mm")

    if not args.apply:
        print("\ndry run. re-run with --apply to rewrite pose.txt "
              "(originals kept as pose_raw.txt)")
        return 0

    for v, meta, src in todo:
        T_target_src = T_target_base @ to_T(*T_BASE_FROM[src])
        t, q, secs, nsecs = parse_pose(v / "pose.txt")
        Tn = T_target_src @ to_T(t, q)
        tn, qn = Tn[:3, 3], R_to_quat(Tn[:3, :3])
        raw = v / "pose_raw.txt"
        if not raw.exists():
            shutil.copy2(v / "pose.txt", raw)
        (v / "pose.txt").write_text(POSE_TXT.format(t=tn, q=qn, secs=secs, nsecs=nsecs))
        meta["frame_normalized_to"] = args.target
        meta["base_frame_original"] = src
        meta["base_frame"] = args.target
        meta["tcp"]["translation"] = tn.tolist()
        meta["tcp"]["quaternion_xyzw"] = qn.tolist()
        (v / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nrewrote {len(todo)} views into {args.target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
