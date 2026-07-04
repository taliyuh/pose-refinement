#!/usr/bin/env python3
import os
import re
import cv2
import numpy as np

# config
RGB_DIR = "/run/host/var/home/bartek/calibration1"
POSES_FILE = "/run/host/var/home/bartek/Documents/robot40human_ws/captured_poses.txt"

TARGET_ID = 2
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# define physical marker size in meters
MARKER_LENGTH = 0.15  

# camera Intrinsics
FX = 750.6503295898438
FY = 750.5856323242188
CX = 643.8098754882812
CY = 363.8945617675781

CAMERA_MATRIX = np.array([
    [FX,  0, CX],
    [ 0, FY, CY],
    [ 0,  0,  1]
], dtype=np.float64)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)

# 3D coordinates of marker corners in its own local object frame
OBJ_POINTS = np.array([
    [-MARKER_LENGTH/2,  MARKER_LENGTH/2, 0],
    [ MARKER_LENGTH/2,  MARKER_LENGTH/2, 0],
    [ MARKER_LENGTH/2, -MARKER_LENGTH/2, 0],
    [-MARKER_LENGTH/2, -MARKER_LENGTH/2, 0]
], dtype=np.float32)


def parse_poses(file_path):
    poses_dict = {}
    pattern = re.compile(
        r"Timestamp:\s+([0-9_]+).*?Position:\s+\[(.*?)\].*?Orientation:\s+\[(.*?)\]"
    )
    if not os.path.exists(file_path):
        print(f"[ERROR] Poses file not found at: {file_path}")
        return poses_dict

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ts_str = match.group(1)
                pos = [float(x) for x in match.group(2).split(',')]
                ori = [float(x) for x in match.group(3).split(',')]
                poses_dict[ts_str] = {"position": pos, "orientation": ori}
    return poses_dict


def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,           1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,           2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
    ])


def main():
    robot_poses = parse_poses(POSES_FILE)
    if not robot_poses:
        return

    rgb_files = [f for f in os.listdir(RGB_DIR) if f.startswith("rgb_") and f.endswith(".png")]
    
    R_gripper2base = []
    t_gripper2base = []
    R_target2camera = []
    t_target2camera = []

    print("[INFO] Processing pairs using perspective PnP geometry...")

    for rgb_name in sorted(rgb_files):
        ts_match = re.search(r"rgb_([0-9_]+)\.png", rgb_name)
        if not ts_match:
            continue
        ts_str = ts_match.group(1)

        if ts_str not in robot_poses:
            continue

        rgb_path = os.path.join(RGB_DIR, rgb_name)
        rgb_img = cv2.imread(rgb_path)
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

        if ids is None or TARGET_ID not in ids:
            continue

        idx = np.where(ids == TARGET_ID)[0][0]
        marker_corners = corners[idx][0]

        # calculate true 3D translation (tvec) and true 3D rotation (rvec) via PnP
        success, rvec, tvec = cv2.solvePnP(
            OBJ_POINTS, 
            marker_corners.astype(np.float32), 
            CAMERA_MATRIX, 
            DIST_COEFFS, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            continue

        # convert rotation vector to a proper 3x3 rotation matrix
        R_target, _ = cv2.Rodrigues(rvec)

        # Get robot data
        pose_data = robot_poses[ts_str]
        t_grip = np.array(pose_data["position"], dtype=np.float64).reshape(3, 1)
        q_grip = pose_data["orientation"] 
        R_grip = quaternion_to_rotation_matrix(q_grip)

        R_gripper2base.append(R_grip)
        t_gripper2base.append(t_grip)
        R_target2camera.append(R_target)
        t_target2camera.append(tvec)

    if len(R_gripper2base) < 3:
        print(f"[ERROR] Not enough valid matched steps. Only found {len(R_gripper2base)}.")
        return

    print(f"[INFO] Computing Hand-Eye calibration using {len(R_gripper2base)} steps...")
    
    # run the calibration solver using the clean data streams
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2camera, t_target2camera,
        method=cv2.CALIB_HAND_EYE_PARK
    )   

    T_cam2gripper = np.eye(4)
    T_cam2gripper[0:3, 0:3] = R_cam2gripper
    T_cam2gripper[0:3, 3:4] = t_cam2gripper

    print("\n" + "="*50)
    print("RESULT: CALIBRATION MATRIX (T_camera_to_hand)")
    print("="*50)
    print(np.array2string(T_cam2gripper, formatter={'float_kind': lambda x: f"{x:10.6f}"}))
    print("="*50)
    print(f"Translation Vector X, Y, Z (meters): {t_cam2gripper.flatten()}")


if __name__ == "__main__":
    main()