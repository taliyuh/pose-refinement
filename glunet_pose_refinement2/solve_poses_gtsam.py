#!/usr/bin/env python3
import os
import sys
import re
import json
import random
from pathlib import Path
import numpy as np
import cv2
import torch
import gtsam
from tqdm import tqdm

# Add directories to system path for imports
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "glunet_src"))
sys.path.insert(0, str(ROOT / "refinement_proposal"))

from models.models_compared import GLU_Net
from pose_lib_improved import (
    solve_corners,
    CORNER_3D,
    S
)
from calib import ARUCO_DICT, ARUCO_PARAMS, ARUCO_TARGET_ID, ARUCO_MARKER_LENGTH, ARUCO_OBJ_POINTS, K as K_720
from aruco_estimate_improved import detect_marker_subpix, plane_from_marker

# Camera setup
K_new = np.array([
    [2251.950927734375, 0.0, 1931.4296875],
    [0.0, 2251.7568359375, 1091.6837158203125],
    [0.0, 0.0, 1.0]
], dtype=np.float32)
D_new = np.array([0.07524467259645462, -0.10655965656042099, -0.00023194379173219204, 0.00031730238697491586, 0.044881995767354965], dtype=np.float32)

V_TIP = np.array([-0.0043625, 0.00081629, 0.34239859])

def load_image(img_path):
    """Load image from disk and undistort it to K_new"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.undistort(img, K_new, D_new, newCameraMatrix=K_new)
    return img

def parse_pose_txt(path):
    """Parse translation and quaternion from pose.txt"""
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
    T[:3, :3] = quaternion_to_rotation_matrix([rx, ry, rz, rw])
    T[:3, 3] = [tx, ty, tz]
    return T

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)

def parse_ground_truth_corners(filepath):
    content = Path(filepath).read_text()
    blocks = re.split(r'(\w+\s+\w+\s+corner):', content)
    corners_data = {}
    for i in range(1, len(blocks), 2):
        name = blocks[i].strip()
        body = blocks[i+1]
        matrix_lines = []
        lines = body.split('\n')
        recording = False
        for line in lines:
            if '- Matrix:' in line:
                recording = True
                continue
            if recording:
                parts = line.strip().split()
                if len(parts) == 4:
                    matrix_lines.append([float(x) for x in parts])
                if len(matrix_lines) == 4:
                    break
        if len(matrix_lines) == 4:
            mat = np.array(matrix_lines)
            corners_data[name] = (mat[:3, :3], mat[:3, 3])
    return corners_data

def get_corners_in_order(corners_data):
    order = ["left top corner", "right top corner", "left bottom corner", "right bottom corner"]
    return [corners_data[name] for name in order]

def kabsch_alignment(P, Q):
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H_mat = P_centered.T @ Q_centered
    U, S_vals, Vt = np.linalg.svd(H_mat)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_Q - R @ centroid_P
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def load_glunet(device):
    """Load pretrained GLU-Net model"""
    model = GLU_Net(
        path_pre_trained_models=str(ROOT / "glunet_src" / "pre_trained_models"),
        model_type="DPED_CityScape_ADE",
        consensus_network=False,
        cyclic_consistency=True,
        iterative_refinement=True,
        apply_flipping_condition=False,
    )
    model.net = model.net.to(device)
    return model

def parse_gt(gt_path):
    """Parse ground truth annotations of board corners in pixel coordinates from ground truth.txt"""
    gt_dict = {}
    if not os.path.exists(gt_path):
        return gt_dict
    content = Path(gt_path).read_text()
    content = re.sub(r'[^\x00-\x7F]+', '', content)
    for line in content.splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        parts = line.split(':', 1)
        name = parts[0].strip()
        pts_str = parts[1].strip()
        if name.startswith('#') or name.lower() == 'ground truth' or name.lower().startswith('order') or "top left" in pts_str.lower():
            continue
        pts = []
        for p in pts_str.split(';'):
            p = p.strip()
            if not p:
                continue
            coords = list(map(float, p.split(',')))
            pts.append(coords)
        if len(pts) == 4:
            gt_dict[name] = np.array(pts, dtype=np.float32)
    top_pattern = r'([a-zA-Z0-9_\u200b\.\-/]+):\s*top left:\s*(\d+),\s*(\d+)\s*top right:\s*(\d+),\s*(\d+)\s*bottom left:\s*(\d+),\s*(\d+)\s*bottom right:\s*(\d+),\s*(\d+)'
    top_matches = re.findall(top_pattern, content, re.MULTILINE)
    for m in top_matches:
        name = m[0]
        pts = [
            [float(m[1]), float(m[2])], # TL
            [float(m[3]), float(m[4])], # TR
            [float(m[5]), float(m[6])], # BL
            [float(m[7]), float(m[8])]  # BR
        ]
        gt_dict[name] = np.array(pts, dtype=np.float32)
    return gt_dict

def undistort_corners(corners):
    """Undistort 2D corners in 4K resolution using K_new and D_new"""
    pts_undistorted = cv2.undistortPoints(
        np.array(corners, dtype=np.float32).reshape(-1, 1, 2),
        K_new,
        D_new,
        P=K_new
    ).reshape(-1, 2)
    return pts_undistorted

def apply_tophat_prep(img, se_sz=25):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_sz, se_sz))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    th_norm = cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.merge([th_norm, th_norm, th_norm])

def dense_matches(model, src, tgt, device, src_corners):
    """Run GLU-Net flow estimation on rectified crop areas to establish correspondences"""
    h_src, w_src = src.shape[:2]
    h_tgt, w_tgt = tgt.shape[:2]
    
    # We will detect the marker in tgt image for ROI crop alignment if possible
    # Tgt corners can be coarsely initialized using ArUco detection on target
    gray_tgt_full = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)
    tgt_marker = detect_marker_subpix(gray_tgt_full)
    
    # Define a canonical 512x400 board coordinate space
    board_canonical = np.float32([[0, 0], [512, 0], [0, 400], [512, 400]])
    
    # We estimate target corners using ArUco marker + blobs pipeline if possible
    # For target crop, if ArUco isn't found or estimation fails, we fallback to resizing
    # Let's write a simple target board estimation for crop:
    tgt_corners = None
    if tgt_marker is not None:
        try:
            # We scale marker corners back to full resolution to find blobs
            # but since detect_marker_subpix returns 720p coords, let's upscale:
            marker_full = tgt_marker / S
            R_aruco, t_aruco = plane_from_marker(tgt_marker)
            
            # Mask out the marker itself inside the target image
            gray_tgt_720 = cv2.resize(gray_tgt_full, (1280, 720))
            mask = np.zeros((720, 1280), np.uint8)
            marker_center = tgt_marker.mean(axis=0)
            marker_sz = np.linalg.norm(tgt_marker[0] - tgt_marker[1])
            x0 = int(marker_center[0] - 1.5 * marker_sz)
            x1 = int(marker_center[0] + 5.0 * marker_sz)
            y0 = int(marker_center[1] - 3.0 * marker_sz)
            y1 = int(marker_center[1] + 3.0 * marker_sz)
            x0, x1 = max(0, x0), min(1280 - 1, x1)
            y0, y1 = max(0, y0), min(720 - 1, y1)
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
            mc = tgt_marker.astype(np.int32)
            pad = int(0.6 * marker_sz)
            mx0, my0 = mc.min(0) - pad
            mx1, my1 = mc.max(0) + pad
            cv2.rectangle(mask, (int(mx0), int(my0)), (int(mx1), int(my1)), 0, -1)
            
            from pose_lib_improved import detect_blobs, estimate, project
            blobs, iscyl = detect_blobs(gray_tgt_720, mask)
            params, cost, rv, tv = estimate(blobs, iscyl, R_aruco, t_aruco, n_th=48, n_uv=17, uv=0.28)
            corners_720 = project(CORNER_3D, rv, tv)
            tgt_corners = corners_720 / S
        except Exception as e:
            pass

    if src_corners is not None and tgt_corners is not None:
        H_rect_src, _ = cv2.findHomography(src_corners, board_canonical)
        H_rect_tgt, _ = cv2.findHomography(tgt_corners, board_canonical)
        
        src_rect = cv2.warpPerspective(src, H_rect_src, (512, 400))
        tgt_rect = cv2.warpPerspective(tgt, H_rect_tgt, (512, 400))
        
        src_prep = apply_tophat_prep(src_rect)
        tgt_prep = apply_tophat_prep(tgt_rect)
        
        with torch.no_grad():
            source_tensor = torch.from_numpy(src_prep).permute(2, 0, 1).unsqueeze(0).to(device)
            target_tensor = torch.from_numpy(tgt_prep).permute(2, 0, 1).unsqueeze(0).to(device)
            flow = model.estimate_flow(source_tensor, target_tensor, device, mode="channel_first")
        flow = flow.squeeze().cpu().numpy()
        
        w_tf, h_tf = tgt_rect.shape[1], tgt_rect.shape[0]
        xx, yy = np.meshgrid(np.arange(w_tf), np.arange(h_tf))
        pts_t = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
        flow_x = cv2.resize(flow[0], (w_tf, h_tf), interpolation=cv2.INTER_LINEAR)
        flow_y = cv2.resize(flow[1], (w_tf, h_tf), interpolation=cv2.INTER_LINEAR)
        pts_s = np.stack((xx + flow_x, yy + flow_y), axis=-1).reshape(-1, 2).astype(np.float32)
        
        # Detect if there's a 180-flip
        H_rect_fit, _ = cv2.findHomography(pts_s, pts_t, cv2.RANSAC, 5.0)
        if H_rect_fit is not None and (H_rect_fit[0, 0] + H_rect_fit[1, 1] < 0):
            tgt_corners = tgt_corners[::-1]
            H_rect_tgt, _ = cv2.findHomography(tgt_corners, board_canonical)
            tgt_rect = cv2.warpPerspective(tgt, H_rect_tgt, (512, 400))
            tgt_prep = apply_tophat_prep(tgt_rect)
            with torch.no_grad():
                target_tensor = torch.from_numpy(tgt_prep).permute(2, 0, 1).unsqueeze(0).to(device)
                flow = model.estimate_flow(source_tensor, target_tensor, device, mode="channel_first")
            flow = flow.squeeze().cpu().numpy()
            flow_x = cv2.resize(flow[0], (w_tf, h_tf), interpolation=cv2.INTER_LINEAR)
            flow_y = cv2.resize(flow[1], (w_tf, h_tf), interpolation=cv2.INTER_LINEAR)
            pts_s = np.stack((xx + flow_x, yy + flow_y), axis=-1).reshape(-1, 2).astype(np.float32)
            
        H_rect_src_inv = np.linalg.inv(H_rect_src)
        H_rect_tgt_inv = np.linalg.inv(H_rect_tgt)
        pts_source = cv2.perspectiveTransform(pts_s.reshape(-1, 1, 2), H_rect_src_inv).reshape(-1, 2)
        pts_target = cv2.perspectiveTransform(pts_t.reshape(-1, 1, 2), H_rect_tgt_inv).reshape(-1, 2)
    else:
        # Direct resize fallback
        max_dim = 1024
        scale_s = max_dim / float(max(h_src, w_src))
        scale_t = max_dim / float(max(h_tgt, w_tgt))
        src_resized = cv2.resize(src, (int(w_src*scale_s), int(h_src*scale_s)))
        tgt_resized = cv2.resize(tgt, (int(w_tgt*scale_t), int(h_tgt*scale_t)))
        
        src_prep = apply_tophat_prep(src_resized)
        tgt_prep = apply_tophat_prep(tgt_resized)
        
        with torch.no_grad():
            source_tensor = torch.from_numpy(src_prep).permute(2, 0, 1).unsqueeze(0).to(device)
            target_tensor = torch.from_numpy(tgt_prep).permute(2, 0, 1).unsqueeze(0).to(device)
            flow = model.estimate_flow(source_tensor, target_tensor, device, mode="channel_first")
        flow = flow.squeeze().cpu().numpy()
        
        w_tf, h_tf = tgt_resized.shape[1], tgt_resized.shape[0]
        xx, yy = np.meshgrid(np.arange(w_tf), np.arange(h_tf))
        pts_target = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
        flow_x = cv2.resize(flow[0], (w_tf, h_tf), interpolation=cv2.INTER_LINEAR)
        flow_y = cv2.resize(flow[1], (w_tf, h_tf), interpolation=cv2.INTER_LINEAR)
        pts_source = np.stack(
            (xx * (scale_t / scale_s) + flow_x / scale_s, yy * (scale_t / scale_s) + flow_y / scale_s),
            axis=-1
        ).reshape(-1, 2).astype(np.float32)
        
    return pts_source, pts_target

def estimate_board_pose(model, src_img, tgt_img, src_corners, device):
    """Estimate board pose in target camera frame using GLU-Net matches and PnP"""
    pts_src, pts_tgt = dense_matches(model, src_img, tgt_img, device, src_corners)
    
    # Filter matches within the board area in the source image
    contour = np.array([src_corners[0], src_corners[1], src_corners[3], src_corners[2]], dtype=np.float32)
    h_src, w_src = src_img.shape[:2]
    mask = np.zeros((h_src, w_src), dtype=np.uint8)
    cv2.fillPoly(mask, [contour.astype(np.int32)], 255)
    
    xs = np.clip(np.round(pts_src[:, 0]).astype(np.int32), 0, w_src - 1)
    ys = np.clip(np.round(pts_src[:, 1]).astype(np.int32), 0, h_src - 1)
    
    inside = mask[ys, xs] > 0
    pts_src_filtered = pts_src[inside]
    pts_tgt_filtered = pts_tgt[inside]
    
    if len(pts_src_filtered) < 10:
        return None
        
    H, h_mask = cv2.findHomography(pts_src_filtered, pts_tgt_filtered, cv2.USAC_MAGSAC, 2.0)
    if H is None:
        return None
        
    pred_corners = cv2.perspectiveTransform(src_corners.reshape(-1, 1, 2), H).reshape(-1, 2)
    pred_corners_720 = pred_corners * S
    rv, tv = solve_corners(pred_corners_720)
    
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(rv)
    T[:3, 3] = tv.ravel()
    return T

# Helper functions for GTSAM conversion
def pose3_from_matrix(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return gtsam.Pose3(gtsam.Rot3(R), t)

def matrix_from_pose3(pose):
    T = np.eye(4)
    T[:3, :3] = pose.rotation().matrix()
    T[:3, 3] = pose.translation()
    return T

def make_hand_eye_factor(C_key, X_key, T_base_EE, noise_model):
    """Custom GTSAM factor for C = T_base_EE * X"""
    def error_func(factor, values, jacobians):
        C = values.atPose3(C_key)
        X = values.atPose3(X_key)
        T_pred = T_base_EE.compose(X)
        err_pose = T_pred.inverse().compose(C)
        err = gtsam.Pose3.Logmap(err_pose)
        
        if jacobians is not None:
            eps = 1e-6
            # Jacobian wrt C
            J_C = np.zeros((6, 6))
            for i in range(6):
                dxi = np.zeros(6)
                dxi[i] = eps
                C_perturbed = C.retract(dxi)
                err_perturbed = gtsam.Pose3.Logmap(T_pred.inverse().compose(C_perturbed))
                J_C[:, i] = (err_perturbed - err) / eps
            jacobians[0] = J_C
            
            # Jacobian wrt X
            J_X = np.zeros((6, 6))
            for i in range(6):
                dxi = np.zeros(6)
                dxi[i] = eps
                X_perturbed = X.retract(dxi)
                T_pred_perturbed = T_base_EE.compose(X_perturbed)
                err_perturbed = gtsam.Pose3.Logmap(T_pred_perturbed.inverse().compose(C))
                J_X[:, i] = (err_perturbed - err) / eps
            jacobians[1] = J_X
            
        return err
        
    return gtsam.CustomFactor(noise_model, [C_key, X_key], error_func)

def main():
    images_dir = ROOT / "images"
    depth_dir = images_dir / "images_depth"
    gt_pixel_path = images_dir / "ground truth.txt"
    gt_pixels = parse_gt(gt_pixel_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading GLU-Net model...")
    model = load_glunet(device)
    print("Model loaded.")
    
    # Dictionary to hold the data: views[pose_idx][view_name] = { 'T_base_tcp', 'img', 'marker_pose_cam', 'board_pose_cam' }
    views = {1: {}, 2: {}, 3: {}}
    
    # 1. Collect views, detect ArUco and run GLU-Net to get raw camera frame estimates
    for p_idx in [1, 2, 3]:
        p_folder = depth_dir / f"pose_{p_idx}"
        subdirs = sorted([d for d in p_folder.iterdir() if d.is_dir()])
        
        # Find which subdir has GT annotations
        annotated_name = None
        for s_dir in subdirs:
            # Check matches in parsed ground truth keys
            # The keys might look like "pose_1/103927_179558"
            match_key = f"pose_{p_idx}/{s_dir.name}"
            if match_key in gt_pixels:
                annotated_name = s_dir.name
                break
                
        if annotated_name is None:
            print(f"[Warning] No annotated image found for pose_{p_idx}! Using first image as fallback.")
            annotated_name = subdirs[0].name
            
        print(f"\nPose {p_idx} - Annotated Source Image: {annotated_name}")
        
        # Load and undistort the source image
        src_path = p_folder / annotated_name / "color.png"
        src_img = load_image(src_path)
        
        # Get undistorted source corners
        match_key = f"pose_{p_idx}/{annotated_name}"
        if match_key in gt_pixels:
            src_corners_distorted = gt_pixels[match_key]
            src_corners = undistort_corners(src_corners_distorted)
        else:
            # Estimate from ArUco if missing GT
            gray_src = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
            marker_src = detect_marker_subpix(gray_src)
            R_src, t_src = plane_from_marker(marker_src)
            mask_src = np.full((720, 1280), 255, np.uint8)
            from pose_lib_improved import detect_blobs, estimate, project
            blobs_src, iscyl_src = detect_blobs(gray_src, mask_src)
            params, cost, rv, tv = estimate(blobs_src, iscyl_src, R_src, t_src)
            src_corners = project(CORNER_3D, rv, tv) / S
            
        # Extract data for all subdirectories
        for s_dir in tqdm(subdirs, desc=f"Processing pose_{p_idx} views"):
            color_path = s_dir / "color.png"
            pose_path = s_dir / "pose.txt"
            
            T_base_tcp = parse_pose_txt(pose_path)
            img = load_image(color_path)
            if img is None:
                continue
                
            # Detect ArUco marker
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            marker = detect_marker_subpix(gray)
            
            T_cam_marker = None
            if marker is not None:
                _, rv, tv = cv2.solvePnP(ARUCO_OBJ_POINTS, marker, K_720, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
                R_marker, _ = cv2.Rodrigues(rv)
                T_cam_marker = np.eye(4)
                T_cam_marker[:3, :3] = R_marker
                T_cam_marker[:3, 3] = tv.ravel()
                # Check physical plausibility of marker pose
                z_trans = T_cam_marker[2, 3]
                z_normal = T_cam_marker[2, 2]
                if not (0.2 < z_trans < 0.9) or z_normal > -0.5:
                    T_cam_marker = None
                    
            # Estimate board pose via GLU-Net
            T_cam_board = None
            if s_dir.name == annotated_name:
                # For the annotated source image, we can run solve_corners directly
                src_corners_720 = src_corners * S
                rv, tv = solve_corners(src_corners_720)
                T_cam_board = np.eye(4)
                T_cam_board[:3, :3], _ = cv2.Rodrigues(rv)
                T_cam_board[:3, 3] = tv.ravel()
            else:
                T_cam_board = estimate_board_pose(model, src_img, img, src_corners, device)
                
            if T_cam_board is not None:
                # Check physical plausibility of board pose
                z_trans = T_cam_board[2, 3]
                z_normal = T_cam_board[2, 2]
                if not (0.2 < z_trans < 0.9) or z_normal > -0.5:
                    T_cam_board = None
                
            views[p_idx][s_dir.name] = {
                'T_base_tcp': T_base_tcp,
                'T_cam_marker': T_cam_marker,
                'T_cam_board': T_cam_board
            }
            
    # 2. Hand-Eye Calibration initialization
    # We collect all frames where ArUco was successfully detected to run calibrateHandEye.
    # Note: cv2.calibrateHandEye assumes target is stationary, which is true within each pose folder,
    # but the target could move between pose folders. Thus, we run it for pose_1 (which has the most views)
    # and use it as our initial seed for optimization.
    R_gripper2base = []
    t_gripper2base = []
    R_target2camera = []
    t_target2camera = []
    
    # We will use pose_1 for the calibration seed
    for name, v in views[1].items():
        if v['T_cam_marker'] is not None:
            # cv2.calibrateHandEye expects gripper to base (T_base_tcp)
            # and target to camera (T_cam_marker)
            R_gripper2base.append(v['T_base_tcp'][:3, :3])
            t_gripper2base.append(v['T_base_tcp'][:3, 3])
            R_target2camera.append(v['T_cam_marker'][:3, :3])
            t_target2camera.append(v['T_cam_marker'][:3, 3])
            
    T_CAM2TCP_seed = np.array([
        [ 0.99649934,  0.06194805, -0.05613828,  0.0112172 ],
        [-0.07473762,  0.96102383, -0.2661719 ,  0.55464933],
        [ 0.0374614,   0.26943577,  0.96228946, -0.17914986],
        [ 0.          , 0.          , 0.          , 1.        ]
    ])
    T_EE2cam_seed = np.linalg.inv(T_CAM2TCP_seed)
    
    print("Initial hand-eye calibration (T_EE2cam) seed:")
    print(T_EE2cam_seed)
    
    # 3. Setup GTSAM Factor Graph joint optimization
    print("\nBuilding GTSAM factor graph...")
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    
    # Key formatting
    # X_key for T_EE_cam (hand-eye calibration)
    X_key = gtsam.symbol('x', 0)
    initial_values.insert(X_key, pose3_from_matrix(T_EE2cam_seed))
    # Hand-eye prior (pin to the calibrateHandEye result with moderate uncertainty)
    graph.add(gtsam.PriorFactorPose3(
        X_key,
        pose3_from_matrix(T_EE2cam_seed),
        gtsam.noiseModel.Diagonal.Sigmas(np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05]))
    ))
    
    # Loop over all pose folders
    for p_idx in [1, 2, 3]:
        # B_key for T_base_board
        B_key = gtsam.symbol('b', p_idx)
        # M_key for T_base_marker
        M_key = gtsam.symbol('m', p_idx)
        
        # Initialize Board and Marker pose using the first frame that has detection
        board_inited = False
        marker_inited = False
        
        # Camera pose keys count
        c_count = 0
        
        # Find first valid frame to initialize board/marker poses
        for name, v in views[p_idx].items():
            T_base_cam = v['T_base_tcp'] @ T_EE2cam_seed
            if v['T_cam_board'] is not None and not board_inited:
                T_base_board_init = T_base_cam @ v['T_cam_board']
                initial_values.insert(B_key, pose3_from_matrix(T_base_board_init))
                board_inited = True
            if v['T_cam_marker'] is not None and not marker_inited:
                T_base_marker_init = T_base_cam @ v['T_cam_marker']
                initial_values.insert(M_key, pose3_from_matrix(T_base_marker_init))
                marker_inited = True
                
        if not board_inited:
            # Fallback initialization
            initial_values.insert(B_key, pose3_from_matrix(np.eye(4)))
        if not marker_inited:
            # Fallback initialization
            initial_values.insert(M_key, pose3_from_matrix(np.eye(4)))
            
        # Add broad prior factors on Board and Marker poses
        graph.add(gtsam.PriorFactorPose3(
            B_key,
            initial_values.atPose3(B_key),
            gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))
        ))
        graph.add(gtsam.PriorFactorPose3(
            M_key,
            initial_values.atPose3(M_key),
            gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))
        ))
        
        # Create factors for all views in this pose folder
        for name, v in views[p_idx].items():
            C_key = gtsam.symbol('c', 100 * p_idx + c_count)
            c_count += 1
            
            # Initial camera pose
            T_base_cam = v['T_base_tcp'] @ T_EE2cam_seed
            initial_values.insert(C_key, pose3_from_matrix(T_base_cam))
            
            # 1. Kinematic constraint: C = T_base_EE * X
            # Standard robot kinematic link has 1 mm position accuracy and 0.1 deg angle accuracy
            kin_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0017, 0.0017, 0.0017, 0.001, 0.001, 0.001]))
            graph.add(make_hand_eye_factor(C_key, X_key, pose3_from_matrix(v['T_base_tcp']), kin_noise))
            
            # 2. ArUco observation factor: Between(C, M, Z_cam_marker)
            if v['T_cam_marker'] is not None:
                # C^-1 * M = Z_cam_marker
                aruco_noise_base = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.017, 0.017, 0.017, 0.005, 0.005, 0.005]))
                aruco_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.0), aruco_noise_base)
                graph.add(gtsam.BetweenFactorPose3(
                    C_key, M_key,
                    pose3_from_matrix(v['T_cam_marker']),
                    aruco_noise
                ))
                
            # 3. GLU-Net board observation factor: Between(C, B, Z_cam_board)
            if v['T_cam_board'] is not None:
                # C^-1 * B = Z_cam_board
                board_noise_base = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0087, 0.0087, 0.0087, 0.003, 0.003, 0.003]))
                board_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.0), board_noise_base)
                graph.add(gtsam.BetweenFactorPose3(
                    C_key, B_key,
                    pose3_from_matrix(v['T_cam_board']),
                    board_noise
                ))
                
    # 4. Optimize
    print(f"\nInitial graph error: {graph.error(initial_values):.4f}")
    print("Running GTSAM optimizer...")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values)
    result = optimizer.optimize()
    print("GTSAM Optimization completed.")
    print(f"Final graph error: {graph.error(result):.4f}")
    
    # Retrieve optimized Hand-Eye pose
    T_EE2cam_opt = matrix_from_pose3(result.atPose3(X_key))
    print("\nOptimized hand-eye calibration (T_EE2cam):")
    print(T_EE2cam_opt)
    
    # 5. Verification against ground truth
    print(f"\n{'='*60}")
    print("VERIFICATION AGAINST TOUCH-PROBE GROUND TRUTH")
    print(f"{'='*60}")
    
    # Dictionary to save results
    optimized_results = {
        "T_EE2cam_opt": T_EE2cam_opt.tolist(),
        "poses": {}
    }
    
    for p_idx in [1, 2, 3]:
        p_folder = depth_dir / f"pose_{p_idx}"
        gt_path = p_folder / "ground_truth.txt"
        
        # Load touch-probe corners in base frame
        c_data = parse_ground_truth_corners(gt_path)
        ordered = get_corners_in_order(c_data)
        pts_base_gt = np.array([R @ V_TIP + t for R, t in ordered])
        
        # Get ground truth board pose via Kabsch
        T_base_board_gt = kabsch_alignment(CORNER_3D, pts_base_gt)
        
        # Retrieve optimized board pose from GTSAM
        B_key = gtsam.symbol('b', p_idx)
        T_base_board_opt = matrix_from_pose3(result.atPose3(B_key))
        
        # Calculate errors
        t_err = np.linalg.norm(T_base_board_opt[:3, 3] - T_base_board_gt[:3, 3]) * 1000.0 # mm
        
        # Rotation error (geodesic angle in degrees)
        R_diff = T_base_board_opt[:3, :3].T @ T_base_board_gt[:3, :3]
        trace_R = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
        r_err = np.arccos(trace_R) * 180.0 / np.pi
        
        # Compute corner translation errors (distance between corresponding 3D corners in base frame)
        corners_opt_base = (T_base_board_opt[:3, :3] @ CORNER_3D.T).T + T_base_board_opt[:3, 3]
        corner_errs = np.linalg.norm(corners_opt_base - pts_base_gt, axis=1) * 1000.0 # mm
        mean_corner_err = np.mean(corner_errs)
        
        print(f"\n--- Board Pose {p_idx} ---")
        print("Optimized Board Pose (T_base_board):")
        print(T_base_board_opt)
        print("Ground Truth Board Pose (Kabsch alignment):")
        print(T_base_board_gt)
        print(f"Translation Error (center): {t_err:.2f} mm")
        print(f"Rotation Error (geodesic): {r_err:.4f} degrees")
        print(f"Mean Corner Distance Error: {mean_corner_err:.2f} mm")
        for idx_c, err_c in enumerate(corner_errs):
            c_name = ["left top", "right top", "left bottom", "right bottom"][idx_c]
            print(f"  - Corner '{c_name}' Error: {err_c:.2f} mm")
            
        optimized_results["poses"][f"pose_{p_idx}"] = {
            "T_base_board_opt": T_base_board_opt.tolist(),
            "T_base_board_gt": T_base_board_gt.tolist(),
            "translation_error_mm": t_err,
            "rotation_error_deg": r_err,
            "mean_corner_error_mm": mean_corner_err,
            "corner_errors_mm": corner_errs.tolist()
        }
        
    # Write optimized results to a json file
    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True)
    results_json = out_dir / "gtsam_optimized_poses.json"
    with open(results_json, 'w') as f:
        json.dump(optimized_results, f, indent=4)
    print(f"\nOptimized results saved to {results_json}")

if __name__ == "__main__":
    main()
