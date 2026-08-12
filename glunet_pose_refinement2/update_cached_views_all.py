#!/usr/bin/env python3
"""
update_cached_views_all.py - Populate cached_views.json for all pose directories (pose_1 to pose_16).
Detects ArUco marker in 4K color images to generate initial T_cam_marker & T_cam_board seed poses,
and parses T_base_tcp from pose.txt.
"""

import sys
import re
import json
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "glunet_src"))
sys.path.insert(0, str(ROOT / "refinement_proposal"))

from calib import K as K_720, DIST as DIST_720, ARUCO_DICT, ARUCO_PARAMS, ARUCO_TARGET_ID, ARUCO_OBJ_POINTS

K_4K = np.array([[2251.950927734375, 0, 1931.4296875],
                 [0, 2251.7568359375, 1091.6837158203125],
                 [0, 0, 1]], np.float64)
DIST_4K = np.array([0.07524467259645462, -0.10655965656042099,
                    -0.00023194379173219204, 0.00031730238697491586,
                    0.044881995767354965], np.float32)

def parse_pose_txt(path):
    with open(path, 'r') as f:
        content = f.read()
    tx = float(re.search(r'translation_x:\s*([-\d.e+]+)', content).group(1))
    ty = float(re.search(r'translation_y:\s*([-\d.e+]+)', content).group(1))
    tz = float(re.search(r'translation_z:\s*([-\d.e+]+)', content).group(1))
    rx = float(re.search(r'rotation_x:\s*([-\d.e+]+)', content).group(1))
    ry = float(re.search(r'rotation_y:\s*([-\d.e+]+)', content).group(1))
    rz = float(re.search(r'rotation_z:\s*([-\d.e+]+)', content).group(1))
    rw = float(re.search(r'rotation_w:\s*([-\d.e+]+)', content).group(1))
    T = np.eye(4)
    x, y, z, w = rx, ry, rz, rw
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
    ])
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T

def detect_marker_pose_4k(gray_full):
    det = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    corners, ids, _ = det.detectMarkers(gray_full)
    if ids is None:
        return None
    ids = ids.ravel()
    for c, i in zip(corners, ids):
        if i == ARUCO_TARGET_ID:
            c_reshaped = c.reshape(-1, 2)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.05)
            refined_c = cv2.cornerSubPix(gray_full, c_reshaped.astype(np.float32), (5, 5), (-1, -1), criteria)
            
            # solvePnP in 4K
            _, rv, tv = cv2.solvePnP(ARUCO_OBJ_POINTS, refined_c, K_4K, DIST_4K, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            R, _ = cv2.Rodrigues(rv)
            t = tv.ravel()
            if np.dot(R[:, 2], -t) < 0:
                R = R @ np.diag([1.0, -1.0, -1.0])
            T_marker = np.eye(4)
            T_marker[:3, :3] = R
            T_marker[:3, 3] = t
            return T_marker
    return None

def main():
    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    cache_file = out_dir / "cached_views.json"

    cache = {}
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cache = json.load(f)

    data_dir = ROOT / "images" / "images_depth"
    pose_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("pose_")])

    print(f"Found {len(pose_dirs)} pose directories in {data_dir.name}...")

    # Marker-to-board offset seed
    # ArUco marker sits approximately at x_board = -0.27m, y_board = 0.0m on the table
    T_marker_to_board = np.eye(4)
    T_marker_to_board[:3, 3] = [0.266, -0.008, 0.0]

    updated_count = 0
    total_views = 0
    missing = []

    for p_dir in pose_dirs:
        p_num = str(int(p_dir.name.split("pose_")[1]))
        if p_num not in cache:
            cache[p_num] = {}

        view_dirs = sorted([d for d in p_dir.iterdir() if d.is_dir()])
        for v_dir in view_dirs:
            view_name = v_dir.name
            total_views += 1

            pose_txt = v_dir / "pose.txt"
            color_png = v_dir / "color.png"

            if not pose_txt.exists() or not color_png.exists():
                continue

            # Load T_base_tcp
            T_base_tcp = parse_pose_txt(pose_txt)

            # Check if view already has cached T_cam_board
            existing_rec = cache[p_num].get(view_name, {})
            T_cam_marker = existing_rec.get("T_cam_marker")
            T_cam_board = existing_rec.get("T_cam_board")

            if T_cam_board is None:
                color_img = cv2.imread(str(color_png))
                if color_img is not None:
                    gray_full = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
                    T_marker_detected = detect_marker_pose_4k(gray_full)

                    if T_marker_detected is not None:
                        T_cam_marker = T_marker_detected.tolist()
                        T_board_est = T_marker_detected @ T_marker_to_board
                        T_cam_board = T_board_est.tolist()
                        updated_count += 1
                    else:
                        # No marker in frame. There used to be a hardcoded
                        # constant pose here, applied silently: it seeded all
                        # 194 views of pose_4..16 identically, every one of
                        # which then failed the mad<1px gate. A wrong seed is
                        # not recoverable downstream -- fit_dense_on_table also
                        # picks the depth ROI from it, so roll/pitch/height are
                        # locked to a plane fitted in the wrong place. Refuse.
                        missing.append(f"pose_{p_num}/{view_name}")
                        continue

            cache[p_num][view_name] = {
                "T_base_tcp": T_base_tcp.tolist(),
                "T_cam_marker": T_cam_marker,
                "T_cam_board": T_cam_board
            }

    if missing:
        by_pose = {}
        for m in missing:
            by_pose[m.split("/")[0]] = by_pose.get(m.split("/")[0], 0) + 1
        raise SystemExit(
            f"\nABORT: no ArUco marker in {len(missing)} views, so no seed exists "
            "for them:\n"
            + "\n".join(f"    {k}: {v} views" for k, v in sorted(by_pose.items()))
            + "\n\nNothing was written. Use the marker-free path instead, which\n"
              "seeds each image from its own rim edges:\n"
              "    .venv/bin/python glunet_pose_refinement2/fit_rims6.py")

    # A seed repeated across views is always a bug -- real poses differ.
    seeds = [np.asarray(r["T_cam_board"])[:3, 3].round(4).tobytes()
             for v in cache.values() for r in v.values()
             if r.get("T_cam_board") is not None]
    if seeds and len(set(seeds)) < 0.5 * len(seeds):
        raise SystemExit(
            f"\nABORT: only {len(set(seeds))} distinct seeds across {len(seeds)} "
            "views. Seeds are being duplicated rather than measured -- the most "
            "likely cause is a stale cached_views.json still holding the constant "
            "fallback pose that this script used to write.")

    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"\nSuccessfully updated {cache_file.name}:")
    print(f"  Total Poses Cached: {len(cache)} ({sorted([int(k) for k in cache.keys()])})")
    print(f"  Total Views Cached: {sum(len(v) for v in cache.values())}")

if __name__ == "__main__":
    main()
