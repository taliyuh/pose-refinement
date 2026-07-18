#!/usr/bin/env python3
import os
import sys
import re
import random
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add directories to system path for imports
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "glunet_src"))
sys.path.insert(0, str(ROOT / "refinement_proposal"))

# Import GLU-Net model
from models.models_compared import GLU_Net

# Import workspace pose estimation library functions
from pose_lib_improved import (
    detect_blobs,
    estimate,
    project,
    solve_corners,
    CORNER_3D,
    MODEL_CENTERS,
    IW,
    IH,
    load_gray
)
from calib import K, DIST, ARUCO_DICT, ARUCO_PARAMS, ARUCO_TARGET_ID
from aruco_estimate_improved import detect_marker_subpix, plane_from_marker

# Full-res dimensions
SW, SH = 3840, 2160
S = np.array([IW / SW, IH / SH])

def detect_marker_fullres(img_full):
    """Detect ArUco marker in full resolution image and apply sub-pixel refinement"""
    gray_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2GRAY)
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
            return refined_c
    return None

def parse_gt(gt_path):
    """Parse ground truth annotations from ground truth.txt"""
    gt_dict = {}
    if not os.path.exists(gt_path):
        return gt_dict
        
    content = Path(gt_path).read_text()
    # Normalize zero-width spaces and non-ascii characters
    content = re.sub(r'[^\x00-\x7F]+', '', content)
    
    for line in content.splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
            
        parts = line.split(':', 1)
        name = parts[0].strip()
        pts_str = parts[1].strip()
        
        # Skip comments or header lines
        if name.startswith('#') or name.lower() == 'ground truth' or name.lower().startswith('order'):
            continue
            
        # Skip if it is the multi-line "top left: ..." format, we'll let the dedicated parser handle it
        if "top left" in pts_str.lower():
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
            
    # Top of the file format (TL, TR, BL, BR explicit text)
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

def find_image_path(img_name, search_dir):
    """Find the path of an image by searching recursively in search_dir"""
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Try finding img_name/color.png
        p = Path(search_dir) / img_name / "color.png"
        if p.is_file():
            return p
        for p in Path(search_dir).rglob(f"*{img_name}*"):
            if p.is_dir():
                color_p = p / "color.png"
                if color_p.is_file():
                    return color_p
    else:
        for p in Path(search_dir).rglob(img_name):
            if p.is_file():
                return p
    return None

def load_image(img_path):
    """Load image from disk and optionally apply camera lens undistortion"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
        
    # === NEW CAMERA UNDISTORTION BLOCK ===
    # Set to False or comment out this block when using the old wood calibration
    UNDISTORT_NEW_CAMERA = True
    if UNDISTORT_NEW_CAMERA and "images_depth" in str(img_path):
        K_new = np.array([
            [2251.950927734375, 0.0, 1931.4296875],
            [0.0, 2251.7568359375, 1091.6837158203125],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        D_new = np.array([0.07524467259645462, -0.10655965656042099, -0.00023194379173219204, 0.00031730238697491586, 0.044881995767354965], dtype=np.float32)
        img = cv2.undistort(img, K_new, D_new, newCameraMatrix=K_new)
    # =====================================
    
    return img

def get_plane_from_depth(img_path):
    """Segment the desk plane from depth.png in camera frame and return R_plane, t_plane"""
    if img_path is None:
        return None, None
        
    img_path = Path(img_path)
    # Find depth.png in the same directory as img_path (e.g. pose_1/103927_179558/depth.png)
    depth_path = img_path.parent / "depth.png"
    if not depth_path.exists():
        return None, None
        
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None, None
        
    # Determine the scaling factor based on the folder path
    path_str = str(img_path)
    scale = 1.0
    if "pose_1" in path_str:
        scale = 0.9510
    elif "pose_2" in path_str:
        scale = 0.9503
    elif "pose_3" in path_str:
        scale = 0.9502
        
    # Segment desk plane
    down = 8
    depth_down = depth[::down, ::down]
    ys, xs = np.where((depth_down > 300) & (depth_down < 1000))
    ds = (depth_down[ys, xs].astype(np.float32) / 1000.0) * scale
    
    fx, fy = 2251.950927734375, 2251.7568359375
    cx, cy = 1931.4296875, 1091.6837158203125
    
    pts_cam = np.column_stack([(xs * down - cx) * ds / fx, (ys * down - cy) * ds / fy, ds])
    if len(pts_cam) < 100:
        return None, None
        
    # Downsample points
    num_pts = len(pts_cam)
    sample_sz = min(num_pts, 3000)
    idx_ds = np.random.choice(num_pts, sample_sz, replace=False)
    pts_cam_ds = pts_cam[idx_ds]
    
    # Run RANSAC plane fitting
    best_inliers = []
    best_plane = None
    max_iters = 200
    threshold = 0.003
    for _ in range(max_iters):
        idx = np.random.choice(sample_sz, 3, replace=False)
        p1, p2, p3 = pts_cam_ds[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n /= norm
        d = -np.dot(n, p1)
        dists = np.abs(np.dot(pts_cam_ds, n) + d)
        inliers = np.where(dists < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (n, d)
            
    if best_plane is None:
        return None, None
        
    # Re-fit with all inliers
    inliers_pts = pts_cam_ds[best_inliers]
    centroid = np.mean(inliers_pts, axis=0)
    centered = inliers_pts - centroid
    _, _, Vh = np.linalg.svd(centered)
    n = Vh[2, :]
    d = -np.dot(n, centroid)
    
    # Ensure normal vector points towards the camera (nz < 0)
    if n[2] > 0:
        n = -n
        d = -d
        
    return n, d

def estimate_corners_from_aruco(img_path):
    """Estimate board corners in full-resolution coordinates using the ArUco marker + blobs pipeline"""
    try:
        bgr_full = load_image(img_path)
        if bgr_full is None:
            return None, None, None
        
        # Get plane normal and offset from depth map if available
        n_ransac, d_ransac = get_plane_from_depth(img_path)
        
        return estimate_corners_from_aruco_img(bgr_full, n_ransac=n_ransac, d_ransac=d_ransac)
    except Exception as e:
        print(f"Error in ArUco estimation for {img_path.name}: {e}")
        return None, None, None

def estimate_corners_from_aruco_img(img_full, n_ransac=None, d_ransac=None):
    """Estimate board corners in full-resolution coordinates from a numpy BGR image array"""
    try:
        gray_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2GRAY)
        bgr = cv2.resize(img_full, (IW, IH))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        marker = detect_marker_subpix(gray_full)
        if marker is None:
            return None, None, None
            
        R_aruco, t_aruco = plane_from_marker(marker)
        
        if n_ransac is not None and d_ransac is not None:
            n = n_ransac
            d = d_ransac
            n_aruco = R_aruco[:, 2]
            
            # Find rotation matrix R_diff that rotates n_aruco to n
            v = np.cross(n_aruco, n)
            s = np.linalg.norm(v)
            c = np.dot(n_aruco, n)
            if s < 1e-6:
                R_diff = np.eye(3)
                if c < 0:
                    R_diff = -np.eye(3)
            else:
                K_skew = np.array([
                    [0.0, -v[2], v[1]],
                    [v[2], 0.0, -v[0]],
                    [-v[1], v[0], 0.0]
                ])
                R_diff = np.eye(3) + K_skew + K_skew @ K_skew * ((1.0 - c) / (s * s))
                
            R_plane = R_diff @ R_aruco
            # Project t_aruco onto the RANSAC plane
            t_plane = t_aruco - (np.dot(n, t_aruco) + d) * n
        else:
            R_plane, t_plane = R_aruco, t_aruco

        # Create ROI mask around the marker where the board is expected
        mask = np.zeros((IH, IW), np.uint8)
        marker_center = marker.mean(axis=0)
        marker_sz = np.linalg.norm(marker[0] - marker[1])
        
        # Bounding box extending around the marker (board is to the right)
        x0 = int(marker_center[0] - 1.5 * marker_sz)
        x1 = int(marker_center[0] + 5.0 * marker_sz)
        y0 = int(marker_center[1] - 3.0 * marker_sz)
        y1 = int(marker_center[1] + 3.0 * marker_sz)
        
        x0, x1 = max(0, x0), min(IW - 1, x1)
        y0, y1 = max(0, y0), min(IH - 1, y1)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
        
        # Mask out the marker itself inside the ROI
        mc = marker.astype(np.int32)
        pad = int(0.6 * marker_sz)
        mx0, my0 = mc.min(0) - pad
        mx1, my1 = mc.max(0) + pad
        cv2.rectangle(mask, (int(mx0), int(my0)), (int(mx1), int(my1)), 0, -1)
        
        blobs, iscyl = detect_blobs(gray, mask)
        params, cost, rv, tv = estimate(blobs, iscyl, R_plane, t_plane,
                                        n_th=48, n_uv=17, uv=0.28)
        
        corners_720 = project(CORNER_3D, rv, tv)
        corners_full = corners_720 / S
        return corners_full, rv, tv
    except Exception as e:
        print(f"Error in ArUco estimation from image array: {e}")
        return None, None, None

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

def apply_tophat_prep(img, se_sz=25):
    """Apply morphological top-hat to highlight bright board features and suppress wood grain"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_sz, se_sz))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    th_norm = cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.merge([th_norm, th_norm, th_norm])

def dense_matches(model, src, tgt, device, src_corners=None, max_dim=1024, src_path=None, tgt_path=None):
    """Run GLU-Net flow estimation on pre-aligned board crops to refine the transformation"""
    h_src, w_src = src.shape[:2]
    h_tgt, w_tgt = tgt.shape[:2]

    # Helper function to run GLU-Net flow estimation on target -> source crops
    def estimate_flow_matches(src_crop, tgt_crop):
        src_prep = apply_tophat_prep(src_crop)
        tgt_prep = apply_tophat_prep(tgt_crop)
        with torch.no_grad():
            source_tensor = torch.from_numpy(src_prep).permute(2, 0, 1).unsqueeze(0)
            target_tensor = torch.from_numpy(tgt_prep).permute(2, 0, 1).unsqueeze(0)
            flow = model.estimate_flow(
                source_tensor,
                target_tensor,
                device,
                mode="channel_first"
            )
        flow = flow.squeeze().cpu().numpy()
        w_tf, h_tf = tgt_crop.shape[1], tgt_crop.shape[0]
        xx, yy = np.meshgrid(np.arange(w_tf), np.arange(h_tf))
        pts_t = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
        flow_x_resized = cv2.resize(flow[0], (w_tf, h_tf), interpolation=cv2.INTER_LINEAR)
        flow_y_resized = cv2.resize(flow[1], (w_tf, h_tf), interpolation=cv2.INTER_LINEAR)
        pts_s = np.stack(
            (xx + flow_x_resized, yy + flow_y_resized),
            axis=-1
        ).reshape(-1, 2).astype(np.float32)
        return pts_s, pts_t

    # 1. Coarsely estimate board corners in both images using ArUco + blobs
    if src_corners is None:
        src_corners, _, _ = estimate_corners_from_aruco(src_path) if src_path is not None else estimate_corners_from_aruco_img(src)
    tgt_corners, _, _ = estimate_corners_from_aruco(tgt_path) if tgt_path is not None else estimate_corners_from_aruco_img(tgt)
    
    use_board_align = (src_corners is not None) and (tgt_corners is not None)
    
    if use_board_align:
        # Define a canonical 512x400 board coordinate space (order: TL, TR, BL, BR)
        board_canonical = np.float32([[0, 0], [512, 0], [0, 400], [512, 400]])
        
        # Compute rectification homographies
        H_rect_src, _ = cv2.findHomography(src_corners, board_canonical)
        H_rect_tgt, _ = cv2.findHomography(tgt_corners, board_canonical)
        
        # Warp the board regions to the canonical space
        src_rect = cv2.warpPerspective(src, H_rect_src, (512, 400))
        tgt_rect = cv2.warpPerspective(tgt, H_rect_tgt, (512, 400))
        
        # Run GLU-Net on the initial pre-alignment crop
        pts_source_flow, pts_target_flow = estimate_flow_matches(src_rect, tgt_rect)
        
        # Fit homography in rectified space to detect 180-degree flip
        H_rect_fit, _ = cv2.findHomography(pts_source_flow, pts_target_flow, cv2.RANSAC, 5.0)
        
        if H_rect_fit is not None and (H_rect_fit[0, 0] + H_rect_fit[1, 1] < 0):
            # 180-degree flip detected! Flip target corners, re-rectify, and re-run matching
            tgt_corners = tgt_corners[::-1]
            H_rect_tgt, _ = cv2.findHomography(tgt_corners, board_canonical)
            tgt_rect = cv2.warpPerspective(tgt, H_rect_tgt, (512, 400))
            pts_source_flow, pts_target_flow = estimate_flow_matches(src_rect, tgt_rect)
            
        H_rect_src_inv = np.linalg.inv(H_rect_src)
        H_rect_tgt_inv = np.linalg.inv(H_rect_tgt)
        
        pts_source = cv2.perspectiveTransform(pts_source_flow.reshape(-1, 1, 2), H_rect_src_inv).reshape(-1, 2)
        pts_target = cv2.perspectiveTransform(pts_target_flow.reshape(-1, 1, 2), H_rect_tgt_inv).reshape(-1, 2)
    else:
        # Fall back to direct matching on resized raw images if coarse detection fails
        if max_dim is not None and max_dim > 0:
            if max(h_src, w_src) > max_dim:
                scale_s = max_dim / float(max(h_src, w_src))
                src_w, src_h = int(w_src * scale_s), int(h_src * scale_s)
            else:
                src_w, src_h = w_src, h_src
                scale_s = 1.0

            h_tgt_flow, w_tgt_flow = tgt.shape[:2]
            if max(h_tgt_flow, w_tgt_flow) > max_dim:
                scale_t = max_dim / float(max(h_tgt_flow, w_tgt_flow))
                tgt_w, tgt_h = int(w_tgt_flow * scale_t), int(h_tgt_flow * scale_t)
            else:
                tgt_w, tgt_h = w_tgt_flow, h_tgt_flow
                scale_t = 1.0
        else:
            src_w, src_h = w_src, h_src
            tgt_w, tgt_h = tgt.shape[1], tgt.shape[0]
            scale_s = 1.0
            scale_t = 1.0

        src_resized = cv2.resize(src, (src_w, src_h))
        tgt_resized = cv2.resize(tgt, (tgt_w, tgt_h))
        
        src_prep = apply_tophat_prep(src_resized)
        tgt_prep = apply_tophat_prep(tgt_resized)
        with torch.no_grad():
            source_tensor = torch.from_numpy(src_prep).permute(2, 0, 1).unsqueeze(0)
            target_tensor = torch.from_numpy(tgt_prep).permute(2, 0, 1).unsqueeze(0)
            flow = model.estimate_flow(
                source_tensor,
                target_tensor,
                device,
                mode="channel_first"
            )
        flow = flow.squeeze().cpu().numpy()
        w_tf, h_tf = tgt_resized.shape[1], tgt_resized.shape[0]
        xx, yy = np.meshgrid(np.arange(w_tf), np.arange(h_tf))
        pts_target_flow = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
        flow_x_resized = cv2.resize(flow[0], (w_tf, h_tf), interpolation=cv2.INTER_LINEAR)
        flow_y_resized = cv2.resize(flow[1], (w_tf, h_tf), interpolation=cv2.INTER_LINEAR)
        pts_source_flow = np.stack(
            (
                xx * (scale_t / scale_s) + flow_x_resized / scale_s,
                yy * (scale_t / scale_s) + flow_y_resized / scale_s
            ),
            axis=-1
        ).reshape(-1, 2).astype(np.float32)
        
        pts_source = pts_source_flow
        pts_target = pts_target_flow

    return pts_source, pts_target

def draw_matches_visual(img_src, img_tgt, pts_src, pts_tgt, max_matches=100):
    """Draw matches connecting source and target keypoints side-by-side"""
    h_s, w_s = img_src.shape[:2]
    h_t, w_t = img_tgt.shape[:2]
    
    # Scale images down for match drawing to fit screen nicely
    draw_h = 720
    scale_s = draw_h / h_s
    scale_t = draw_h / h_t
    
    img_src_d = cv2.resize(img_src, (int(w_s * scale_s), draw_h))
    img_tgt_d = cv2.resize(img_tgt, (int(w_t * scale_t), draw_h))
    
    canvas = np.hstack([img_src_d, img_tgt_d])
    offset_x = img_src_d.shape[1]
    
    if len(pts_src) > max_matches:
        indices = np.random.choice(len(pts_src), max_matches, replace=False)
        pts_s_draw = pts_src[indices]
        pts_t_draw = pts_tgt[indices]
    else:
        pts_s_draw = pts_src
        pts_t_draw = pts_tgt
        
    for p_s, p_t in zip(pts_s_draw, pts_t_draw):
        pt1 = (int(p_s[0] * scale_s), int(p_s[1] * scale_s))
        pt2 = (int(p_t[0] * scale_t) + offset_x, int(p_t[1] * scale_t))
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)
        
    return canvas

def lookup_gt(path, gt):
    """Retrieve ground-truth corners for a file using exact name or path substring match"""
    path_str = str(path)
    if path.name in gt:
        return gt[path.name]
    for key, val in gt.items():
        if key in path_str:
            return val
    return None

def evaluate_pair(model, src_path, tgt_path, gt, device, out_dir):
    """Evaluate a single image pair, estimate corners using GLU-Net, and save visualization"""
    src_name = src_path.name
    tgt_name = tgt_path.name
    
    # 1. Load ground truth corners or fall back to ArUco
    src_corners_gt = lookup_gt(src_path, gt)
    tgt_corners_gt = lookup_gt(tgt_path, gt)
    
    src_corners = src_corners_gt
    src_method = "Ground Truth"
    
    if src_corners is None:
        src_corners, _, _ = estimate_corners_from_aruco(src_path)
        src_method = "ArUco Estimate"
        
    if src_corners is None:
        print(f"[Warning] Could not get source corners for {src_name} (no GT and no ArUco detected). Skipping pair.")
        return None
        
    # 2. Get dense matches from GLU-Net
    img_src = load_image(src_path)
    img_tgt = load_image(tgt_path)
    if img_src is None or img_tgt is None:
        print(f"[Warning] Failed to load images {src_name} or {tgt_name}.")
        return None
        
    h_src, w_src = img_src.shape[:2]
    h_tgt, w_tgt = img_tgt.shape[:2]
    
    pts_src, pts_tgt = dense_matches(model, img_src, img_tgt, device, src_corners=src_corners, src_path=src_path, tgt_path=tgt_path)
    
    # 3. Filter matches using the source board mask
    # Bounding contour: TL -> TR -> BR -> BL
    contour = np.array([src_corners[0], src_corners[1], src_corners[3], src_corners[2]], dtype=np.float32)
    
    # Fast masking in NumPy
    mask = np.zeros((h_src, w_src), dtype=np.uint8)
    cv2.fillPoly(mask, [contour.astype(np.int32)], 255)
    
    xs = np.clip(np.round(pts_src[:, 0]).astype(np.int32), 0, w_src - 1)
    ys = np.clip(np.round(pts_src[:, 1]).astype(np.int32), 0, h_src - 1)
    
    inside = mask[ys, xs] > 0
    pts_src_filtered = pts_src[inside]
    pts_tgt_filtered = pts_tgt[inside]
    
    if len(pts_src_filtered) < 10:
        print(f"[Warning] Too few matches found inside the board for pair {src_name} -> {tgt_name}. Skipping.")
        return None
        
    # 4. Compute homography H from matches
    H, h_mask = cv2.findHomography(pts_src_filtered, pts_tgt_filtered, cv2.RANSAC, 5.0)
    if H is None:
        print(f"[Warning] Homography estimation failed for {src_name} -> {tgt_name}.")
        return None
        
    # Filter matches further by RANSAC inliers
    inliers = h_mask.ravel() > 0
    pts_src_inliers = pts_src_filtered[inliers]
    pts_tgt_inliers = pts_tgt_filtered[inliers]
    
    # 5. Predict target corners
    pred_corners = cv2.perspectiveTransform(src_corners.reshape(-1, 1, 2), H).reshape(-1, 2)
    
    # 6. Quantitative Error
    err = None
    if tgt_corners_gt is not None:
        err = np.linalg.norm(pred_corners - tgt_corners_gt, axis=1).mean()
        
    # 7. Solve 3D target pose from predicted corners
    pred_corners_720 = pred_corners * S
    rv, tv = solve_corners(pred_corners_720)
    
    # Project model shape centers
    proj_centers_720 = project(MODEL_CENTERS, rv, tv)
    proj_centers_full = proj_centers_720 / S
    
    # Project board corners
    proj_corners_720_board = project(CORNER_3D, rv, tv)
    proj_corners_full_board = proj_corners_720_board / S
    
    # ArUco estimate on target (for visual comparison)
    tgt_corners_aruco, _, _ = estimate_corners_from_aruco(tgt_path)
    
    # 8. Visualizations
    # Image 1: Source with corners
    vis_src = img_src.copy()
    cv2.polylines(vis_src, [contour.astype(np.int32)], True, (255, 0, 0), 4, cv2.LINE_AA) # Blue
    for idx, (x, y) in enumerate(src_corners):
        cv2.circle(vis_src, (int(x), int(y)), 10, (0, 255, 0), -1)
        cv2.putText(vis_src, f"{idx}", (int(x)+15, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
    cv2.putText(vis_src, f"Source ({src_method})", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 8)
    vis_src = cv2.resize(vis_src, (960, 540))
    
    # Image 2: Target with predictions and shape overlays
    vis_tgt = img_tgt.copy()
    
    # Draw Ground Truth in Red
    if tgt_corners_gt is not None:
        cv2.polylines(vis_tgt, [np.array([tgt_corners_gt[0], tgt_corners_gt[1], tgt_corners_gt[3], tgt_corners_gt[2]], dtype=np.int32)], True, (0, 0, 255), 5, cv2.LINE_AA)
        
    # Draw GLU-Net predicted board contour in Green (projected from estimated pose)
    cv2.polylines(vis_tgt, [np.array([proj_corners_full_board[0], proj_corners_full_board[1], proj_corners_full_board[3], proj_corners_full_board[2]], dtype=np.int32)], True, (0, 255, 0), 4, cv2.LINE_AA)
    
    # Draw ArUco-detected corners in Magenta if available
    if tgt_corners_aruco is not None:
        cv2.polylines(vis_tgt, [np.array([tgt_corners_aruco[0], tgt_corners_aruco[1], tgt_corners_aruco[3], tgt_corners_aruco[2]], dtype=np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
        
    # Draw projected shape centers in yellow
    for pt in proj_centers_full:
        cv2.circle(vis_tgt, (int(pt[0]), int(pt[1])), 10, (0, 255, 255), 3, cv2.LINE_AA)
        
    # Text annotation
    err_text = f"Corner Err: {err:.2f}px" if err is not None else "No GT"
    cv2.putText(vis_tgt, f"Target: {err_text}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 8)
    vis_tgt = cv2.resize(vis_tgt, (960, 540))
    
    # Image 3: Blended Warped Source + Target
    warped_src = cv2.warpPerspective(img_src, H, (w_tgt, h_tgt))
    blended = cv2.addWeighted(img_tgt, 0.5, warped_src, 0.5, 0)
    cv2.putText(blended, "Blended Overlay (Warped Src + Tgt)", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 8)
    vis_blended = cv2.resize(blended, (960, 540))
    
    # Image 4: Matching keypoints line drawing
    match_canvas = draw_matches_visual(img_src, img_tgt, pts_src_inliers, pts_tgt_inliers, max_matches=100)
    cv2.putText(match_canvas, f"GLU-Net Inliers: {len(pts_src_inliers)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
    vis_matches = cv2.resize(match_canvas, (960, 540))
    
    # Combine all 4 into a 2x2 grid
    top_row = np.hstack([vis_src, vis_tgt])
    bottom_row = np.hstack([vis_blended, vis_matches])
    collage = np.vstack([top_row, bottom_row])
    
    # Save the collage
    out_path = out_dir / f"match_{Path(src_name).stem}_to_{Path(tgt_name).stem}.png"
    cv2.imwrite(str(out_path), collage)
    
    init_err = None
    if tgt_corners_aruco is not None and tgt_corners_gt is not None:
        init_err = np.linalg.norm(tgt_corners_aruco - tgt_corners_gt, axis=1).mean()
        
    print(f"Processed: {src_name} -> {tgt_name} | Matches: {len(pts_src_inliers)} | GLU-Net Corner Err: {f'{err:.2f} px' if err is not None else 'N/A'} | Initial constrained Err: {f'{init_err:.2f} px' if init_err is not None else 'N/A'}")
    return err

def get_dataset_and_pose(path):
    """Determine dataset (wood/depth) and pose name (pose_1/2/3) from image path"""
    parts = path.parts
    if "images_wood" in parts:
        idx = parts.index("images_wood")
        pose = parts[idx + 1]
        return "wood", pose
    elif "images_depth" in parts:
        idx = parts.index("images_depth")
        pose = parts[idx + 1]
        return "depth", pose
    return None, None

def main():
    parser = argparse.ArgumentParser(description="Estimate board transformation between two photos using GLU-Net")
    parser.add_argument("--source", type=str, default=None, help="Path or filename of source image")
    parser.add_argument("--target", type=str, default=None, help="Path or filename of target image")
    parser.add_argument("--evaluate-all", action="store_true", help="Evaluate all pairs of annotated images")
    parser.add_argument("--max-dim", type=int, default=1024, help="Resize max dimension for GLU-Net OOM prevention")
    args = parser.parse_args()

    # Setup directories
    images_dir = ROOT / "images"
    gt_path = ROOT / "images" / "ground truth.txt"
    out_dir = HERE / "out"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Parse ground truth
    gt = parse_gt(gt_path)
    
    # Correct the typo in image_115740_590015.png BR corner: change 1685 to 2685
    if "image_115740_590015.png" in gt:
        gt["image_115740_590015.png"][3, 0] = 2685.0

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading GLU-Net pretrained model...")
    model = load_glunet(device)
    print("Model loaded successfully.")

    if args.evaluate_all:
        print("Evaluating all pairs of annotated images...")
        annotated_images = list(gt.keys())
        
        # Find paths for all annotated images
        annotated_paths = {}
        for name in annotated_images:
            p = find_image_path(name, images_dir)
            if p is not None:
                annotated_paths[name] = p
                
        valid_names = list(annotated_paths.keys())
        print(f"Found {len(valid_names)} annotated images out of {len(annotated_images)}")
        
        # Form all unique ordered pairs (excluding self-pairs, same dataset/pose only)
        pairs = []
        for i, name_s in enumerate(valid_names):
            for j, name_t in enumerate(valid_names):
                if i != j:
                    path_s = annotated_paths[name_s]
                    path_t = annotated_paths[name_t]
                    ds_s, pose_s = get_dataset_and_pose(path_s)
                    ds_t, pose_t = get_dataset_and_pose(path_t)
                    if ds_s is not None and ds_s == ds_t and pose_s == pose_t:
                        pairs.append((path_s, path_t))
                        
        print(f"Filtered pairs (same dataset and pose folder) to evaluate: {len(pairs)}")
        
        errors = []
        for src_path, tgt_path in tqdm(pairs):
            err = evaluate_pair(model, src_path, tgt_path, gt, device, out_dir)
            if err is not None:
                errors.append(err)
                
        if errors:
            print(f"\nAverage Mean Corner Distance Error over {len(errors)} pairs: {np.mean(errors):.2f} px (full-res)")
        else:
            print("\nNo pairs successfully evaluated.")
            
    else:
        # Single pair mode
        # If paths are not specified, select two random annotated images
        annotated_images = list(gt.keys())
        annotated_paths = {}
        for name in annotated_images:
            p = find_image_path(name, images_dir)
            if p is not None:
                annotated_paths[name] = p
                
        valid_names = list(annotated_paths.keys())
        
        if args.source is None or args.target is None:
            if len(valid_names) < 2:
                print("[Error] Not enough annotated images to select a random pair.")
                return
            src_name, tgt_name = random.sample(valid_names, 2)
            src_path = annotated_paths[src_name]
            tgt_path = annotated_paths[tgt_name]
            print(f"Randomly selected annotated pair: {src_name} -> {tgt_name}")
        else:
            # Check if arguments are paths or filenames
            src_path = Path(args.source)
            if not src_path.exists():
                src_path = find_image_path(args.source, images_dir)
                
            tgt_path = Path(args.target)
            if not tgt_path.exists():
                tgt_path = find_image_path(args.target, images_dir)
                
            if src_path is None or not src_path.exists() or tgt_path is None or not tgt_path.exists():
                print(f"[Error] Could not locate source/target images. Source: {args.source}, Target: {args.target}")
                return
                
        evaluate_pair(model, src_path, tgt_path, gt, device, out_dir)

if __name__ == "__main__":
    main()
