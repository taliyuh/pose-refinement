#!/usr/bin/env python3
"""
calib.py — Single source of truth for all constants and shared helpers.

Imported by both BlenderProc processes (generate_glunet_data.py) and torch
processes (pose_estimate_glunet.py, finetune_glunet.py, aruco_localise.py,
pose_estimate.py, visualise_pose.py, check_angles.py, check_aruco_consistency.py).

All values here are numpy‑only; no BlenderProc, torch, or gtsam imports.
"""

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Camera intrinsics (identical across all real‑camera processes)
# ---------------------------------------------------------------------------
FX: float = 750.6503295898438
FY: float = 750.5856323242188
CX: float = 643.8098754882812
CY: float = 363.8945617675781

K: np.ndarray = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

DIST: np.ndarray = np.zeros((4, 1), dtype=np.float64)

# Real image size (all real captures, templates rendered at same size)
IMG_W: int = 1280
IMG_H: int = 720

# Crop size for GLU-Net inference (square, board-filling)
CROP_SIZE: int = 512

# ---------------------------------------------------------------------------
# Hand‑eye calibration: camera → TCP (gripper) transform
# ---------------------------------------------------------------------------
T_CAM2TCP: np.ndarray = np.array([
    [-0.999079, -0.003930,  0.042726, -0.028779],
    [-0.029594, -0.657899, -0.752524,  0.634149],
    [ 0.031067, -0.753096,  0.657177,  0.328725],
    [ 0.000000,  0.000000,  0.000000,  1.000000],
], dtype=np.float64)

T_TCP2CAM: np.ndarray = np.linalg.inv(T_CAM2TCP)

# ---------------------------------------------------------------------------
# Approximate board pose in robot base frame (hand‑measured prior)
# ---------------------------------------------------------------------------
T_BASE_BOARD_APPROX: np.ndarray = np.array([
    [-0.223914, -0.226892,  0.947831,  0.591981],
    [ 0.330123, -0.932691, -0.145280,  0.056846],
    [ 0.916996,  0.280370,  0.283745,  0.788421],
    [ 0.000000,  0.000000,  0.000000,  1.000000],
], dtype=np.float64)

# ---------------------------------------------------------------------------
# STL / board geometry (meters)
# ---------------------------------------------------------------------------
STL_SCALE: float = 0.006          # mm → m for the STL file
BOARD_Z_OFFSET: float = 0.005     # vertical offset of the board from world Z=0

# Board axis-aligned bounding box in its local (board) frame.
# Derived from the 200×160×20 mm board dimensions.
BOARD_AABB_BOARD_FRAME: dict = {
    "x_min": -0.10, "x_max":  0.10,
    "y_min": -0.08, "y_max":  0.08,
    "z_min":  0.00, "z_max":  0.02,
}

# 8 corners of the board bounding box (local frame)
BOARD_CORNERS_LOCAL: np.ndarray = np.array([
    [-0.10, -0.08,  0.00],
    [ 0.10, -0.08,  0.00],
    [ 0.10,  0.08,  0.00],
    [-0.10,  0.08,  0.00],
    [-0.10, -0.08,  0.02],
    [ 0.10, -0.08,  0.02],
    [ 0.10,  0.08,  0.02],
    [-0.10,  0.08,  0.02],
], dtype=np.float32)

# ---------------------------------------------------------------------------
# ArUco
# ---------------------------------------------------------------------------
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

ARUCO_TARGET_ID: int = 2
ARUCO_MARKER_LENGTH: float = 0.106  # meters

ARUCO_OBJ_POINTS: np.ndarray = np.array([
    [-ARUCO_MARKER_LENGTH / 2,  ARUCO_MARKER_LENGTH / 2, 0.0],
    [ ARUCO_MARKER_LENGTH / 2,  ARUCO_MARKER_LENGTH / 2, 0.0],
    [ ARUCO_MARKER_LENGTH / 2, -ARUCO_MARKER_LENGTH / 2, 0.0],
    [-ARUCO_MARKER_LENGTH / 2, -ARUCO_MARKER_LENGTH / 2, 0.0],
], dtype=np.float32)

# ---------------------------------------------------------------------------
# Per‑frame acceptance gate thresholds (Phase 3 step 7)
# ---------------------------------------------------------------------------
ACCEPTANCE_GATE: dict = {
    "min_inliers": 30,
    "min_inlier_ratio": 0.25,
    "max_mean_reproj_px": 1.5,
    "max_angle_prior_deg": 15.0,
    "max_translation_prior_m": 0.08,
}

# ---------------------------------------------------------------------------
# PnP configuration
# ---------------------------------------------------------------------------
PNP_ITERATIONS: int = 5000
PNP_REPROJ_ERR: float = 2.5
PNP_CONFIDENCE: float = 0.999

# ---------------------------------------------------------------------------
# Correspondence filter stack thresholds (pixels in CROP_SIZE coords)
# ---------------------------------------------------------------------------
FB_CONSISTENCY_THRESH: float = 1.5   # forward-backward consistency (px)
DEPTH_DISCONTINUITY_THRESH: float = 0.005  # Sobel gradient magnitude (meters/px)
PRIOR_REPROJ_GATE_PX: float = 40.0   # prior gating (px in original coords)
FLOW_COHERENCE_WINDOW: int = 7        # local flow coherence window
FLOW_COHERENCE_THRESH: float = 3.0    # deviation from median (px)

# ---------------------------------------------------------------------------
# GLU-Net fine‑tuning defaults
# ---------------------------------------------------------------------------
FINETUNE_LR: float = 1e-5
FINETUNE_WD: float = 4e-4
FINETUNE_BATCH_SIZE: int = 8
FINETUNE_EPOCHS: int = 30
FINETUNE_MILESTONES: tuple = (15, 25)
FINETUNE_LOSS_WEIGHTS: tuple = (0.32, 0.08, 0.02, 0.01)
FINETUNE_VAL_SPLIT: float = 0.05

# ---------------------------------------------------------------------------
# Helper functions — pose / quaternion / projection
# ---------------------------------------------------------------------------

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """q = [qx, qy, qz, qw] → 3×3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → [qx, qy, qz, qw]."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def pose_to_matrix(pos: np.ndarray, ori_xyzw: np.ndarray) -> np.ndarray:
    """[x,y,z], [qx,qy,qz,qw] → 4×4 transform."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_rotation_matrix(ori_xyzw)
    T[:3,  3] = pos
    return T


def matrix_to_pose(T: np.ndarray) -> tuple:
    """4×4 → (pos_xyz, quat_xyzw)."""
    pos = T[:3, 3].copy()
    quat = rotation_matrix_to_quaternion(T[:3, :3])
    return pos, quat


# Blender ↔ OpenCV camera coordinate flip
FLIP_BLENDER_OPENCV: np.ndarray = np.diag([1.0, -1.0, -1.0])
# (Blender has +Z backward, +Y up; OpenCV has +Z forward, -Y up)


def project(K_mat: np.ndarray, T_w2c: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
    """Project 3D world points (N×3) to 2D image points (N×2) using K and T_w2c."""
    N = pts_world.shape[0]
    pts_h = np.hstack([pts_world, np.ones((N, 1), dtype=np.float64)])
    pts_cam = (T_w2c @ pts_h.T).T[:, :3]  # N×3 in camera coords
    uvw = pts_cam @ K_mat.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv


def backproject_blender(K_mat: np.ndarray, depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """
    Blender‑frame back‑projection.
    Given depth d at pixel (u, v) in Blender convention (+Y down, +Z into scene):
        x = (u - cx) * d / fx
        y = -(v - cy) * d / fy   (Blender Y flips relative to OpenCV)
        z = -d
    Returns N×3 points in Blender camera frame.
    """
    fx, fy = K_mat[0, 0], K_mat[1, 1]
    cx, cy = K_mat[0, 2], K_mat[1, 2]
    u, v = uv[:, 0], uv[:, 1]
    d = depth
    x = (u - cx) * d / fx
    y = -(v - cy) * d / fy
    z = -d
    return np.column_stack([x, y, z])


def backproject_blender_single(K_mat, depth_val, u, v):
    """Single‑point version."""
    fx, fy = K_mat[0, 0], K_mat[1, 1]
    cx, cy = K_mat[0, 2], K_mat[1, 2]
    x = (u - cx) * depth_val / fx
    y = -(v - cy) * depth_val / fy
    z = -depth_val
    return np.array([x, y, z])


# ---------------------------------------------------------------------------
# Crop-transform helpers
# ---------------------------------------------------------------------------

def crop_bbox(uv: np.ndarray, pad_factor: float = 1.3) -> tuple:
    """
    Given N×2 image points, compute square ROI bounding box.
    Returns (x0, y0, w, h) in original image coords.
    """
    xmin, ymin = uv.min(axis=0)
    xmax, ymax = uv.max(axis=0)
    cx_c, cy_c = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    half = max(xmax - xmin, ymax - ymin) * pad_factor / 2.0
    half = max(half, 10.0)
    x0 = max(0.0, cx_c - half)
    y0 = max(0.0, cy_c - half)
    w = min(IMG_W - x0, 2.0 * half)
    h = min(IMG_H - y0, 2.0 * half)
    side = min(w, h)
    x0 = max(0.0, cx_c - side / 2.0)
    y0 = max(0.0, cy_c - side / 2.0)
    side = min(IMG_W - x0, IMG_H - y0, side)
    return (int(x0), int(y0), int(side), int(side))


def crop_resize(img: np.ndarray, crop: tuple, out_size: int = CROP_SIZE) -> np.ndarray:
    """
    Crop and resize an image.
    crop = (x0, y0, w, h) in original coords → out_size × out_size.
    """
    x0, y0, w, h = crop
    roi = img[y0:y0+h, x0:x0+w]
    return cv2.resize(roi, (out_size, out_size), interpolation=cv2.INTER_AREA)


def crop_transform(x0: int, y0: int, w: int, h: int, out_size: int = CROP_SIZE) -> callable:
    """Returns a function mapping (u,v) from original coords to (u',v') in crop coords."""
    def _map(uv_in):
        u = uv_in[..., 0]
        v = uv_in[..., 1]
        u_out = (u - x0) * out_size / w
        v_out = (v - y0) * out_size / h
        return np.stack([u_out, v_out], axis=-1)
    return _map


def crop_transform_inverse(x0: int, y0: int, w: int, h: int, out_size: int = CROP_SIZE) -> callable:
    """Inverse: (u',v') in crop coords → (u,v) in original coords."""
    def _map(uv_crop):
        uc = uv_crop[..., 0]
        vc = uv_crop[..., 1]
        u_out = uc * w / out_size + x0
        v_out = vc * h / out_size + y0
        return np.stack([u_out, v_out], axis=-1)
    return _map


# ---------------------------------------------------------------------------
# Pose comparison helpers
# ---------------------------------------------------------------------------

def geodesic_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angular distance between two 3×3 rotation matrices, in degrees."""
    R_diff = R1.T @ R2
    tr = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(tr) * 180.0 / np.pi


def translation_error_m(T1: np.ndarray, T2: np.ndarray) -> float:
    """Euclidean distance between translations of two 4×4 transforms."""
    return float(np.linalg.norm(T1[:3, 3] - T2[:3, 3]))


def camera_view_direction(T_cam2world: np.ndarray, blender: bool = False) -> np.ndarray:
    """
    Return the camera's optical axis direction in world frame.
    In OpenCV convention: +Z; in Blender convention: -Z.
    """
    sign = -1.0 if blender else 1.0
    v = sign * T_cam2world[:3, 2].copy()
    return v / np.linalg.norm(v)
