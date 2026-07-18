import os
import re
import cv2
import sys
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

# Add workspace to path
HERE = Path("/var/home/bartek/board_pose_obtain/glunet_pose_refinement2")
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "refinement_proposal"))

from pose_lib_improved import (
    detect_blobs,
    estimate,
    project,
    CORNER_3D,
    IW,
    IH,
    S
)

# Intrinsics
K = np.array([
    [750.6503295898438, 0.0, 643.8098754882812],
    [0.0, 750.5856323242188, 363.8945617675781],
    [0.0, 0.0, 1.0]
])
SW, SH = 3840, 2160
K_full = K.copy()
K_full[0, 0] *= SW / 1280.0
K_full[1, 1] *= SH / 720.0
K_full[0, 2] *= SW / 1280.0
K_full[1, 2] *= SH / 720.0

V_TIP = np.array([-0.0043625, 0.00081629, 0.34239859])

def rotation_vector_to_matrix(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return R

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)

def parse_pose_txt(path):
    with open(path, 'r') as f:
        content = f.read()
    tx = float(re.search(r'translation_x:\s*([-\d.]+)', content).group(1))
    ty = float(re.search(r'translation_y:\s*([-\d.]+)', content).group(1))
    tz = float(re.search(r'translation_z:\s*([-\d.]+)', content).group(1))
    rx = float(re.search(r'rotation_x:\s*([-\d.]+)', content).group(1))
    ry = float(re.search(r'rotation_y:\s*([-\d.]+)', content).group(1))
    rz = float(re.search(r'rotation_z:\s*([-\d.]+)', content).group(1))
    rw = float(re.search(r'rotation_w:\s*([-\d.]+)', content).group(1))
    T = np.eye(4)
    T[:3, :3] = quaternion_to_rotation_matrix([rx, ry, rz, rw])
    T[:3, 3] = [tx, ty, tz]
    return T

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
    H = P_centered.T @ Q_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_Q - R @ centroid_P
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def fit_plane_ransac(pts, threshold=0.003, max_iters=200):
    best_inliers = []
    best_plane = None
    num_pts = len(pts)
    if num_pts < 3:
        return None, None
        
    for _ in range(max_iters):
        idx = np.random.choice(num_pts, 3, replace=False)
        p1, p2, p3 = pts[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n /= norm
        d = -np.dot(n, p1)
        dists = np.abs(np.dot(pts, n) + d)
        inliers = np.where(dists < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (n, d)
            
    if best_plane is None:
        return None, None
        
    # Re-fit using all inliers
    inliers_pts = pts[best_inliers]
    centroid = np.mean(inliers_pts, axis=0)
    centered = inliers_pts - centroid
    _, _, Vh = np.linalg.svd(centered)
    n = Vh[2, :]
    d = -np.dot(n, centroid)
    if n[2] < 0:
        n = -n
        d = -d
    return (n, d), best_inliers

def main():
    base_path = Path("/var/home/bartek/board_pose_obtain")
    depth_dir = base_path / "images" / "images_depth"
    out_dir = base_path / "glunet_pose_refinement2" / "out"
    out_dir.mkdir(exist_ok=True)
    
    # Hand-eye seed from Pose 1 plane optimization
    T_CAM2TCP_seed = np.array([
        [ 0.99649934,  0.06194805, -0.05613828,  0.0112172 ],
        [-0.07473762,  0.96102383, -0.2661719 ,  0.55464933],
        [ 0.0374614,   0.26943577,  0.96228946, -0.17914986],
        [ 0.          , 0.          , 0.          , 1.        ]
    ])
    T_TCP2CAM_seed = np.linalg.inv(T_CAM2TCP_seed)
    
    r_init, _ = cv2.Rodrigues(T_TCP2CAM_seed[:3, :3])
    t_init = T_TCP2CAM_seed[:3, 3]
    x0 = np.hstack([r_init.ravel(), t_init])
    
    for p_folder in sorted(depth_dir.glob("pose_*")):
        gt_path = p_folder / "ground_truth.txt"
        if not gt_path.exists():
            continue
            
        print(f"\n==================== PROCESSING {p_folder.name} ====================")
        
        # 1. Load ground truth board pose as coordinate prior
        c_data = parse_ground_truth_corners(gt_path)
        ordered = get_corners_in_order(c_data)
        pts_base = np.array([R @ V_TIP + t for R, t in ordered])
        T_base_board_gt = kabsch_alignment(CORNER_3D, pts_base)
        
        depth_scales = {
            "pose_1": 0.9510,
            "pose_2": 0.9503,
            "pose_3": 0.9502
        }
        scale = depth_scales.get(p_folder.name, 1.0)
        
        # 2. Gather views and run per-view RANSAC plane segmentation in camera frame
        subdirs = sorted([d for d in p_folder.iterdir() if d.is_dir()])
        views = []
        
        for s_dir in subdirs:
            depth_path = s_dir / "depth.png"
            pose_path = s_dir / "pose.txt"
            
            T_base_tcp = parse_pose_txt(pose_path)
            
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
                
            down = 8
            depth_down = depth[::down, ::down]
            ys, xs = np.where((depth_down > 300) & (depth_down < 1000))
            ds = (depth_down[ys, xs].astype(np.float32) / 1000.0) * scale
            
            fx, fy = K_full[0, 0], K_full[1, 1]
            cx, cy = K_full[0, 2], K_full[1, 2]
            pts_cam = np.column_stack([(xs * down - cx) * ds / fx, (ys * down - cy) * ds / fy, ds])
            
            if len(pts_cam) < 3000:
                continue
                
            # Downsample BEFORE RANSAC to avoid OOM and speed up
            idx_ds = np.random.choice(len(pts_cam), 3000, replace=False)
            pts_cam_ds = pts_cam[idx_ds]
            
            # Segment desk points using RANSAC
            plane, inliers = fit_plane_ransac(pts_cam_ds, threshold=0.003)
            if plane is None or len(inliers) < 1000:
                continue
                
            pts_cam_filtered = pts_cam_ds[inliers]
            
            # Subsample for speed during hand-eye optimization
            if len(pts_cam_filtered) > 500:
                idx = np.random.choice(len(pts_cam_filtered), 500, replace=False)
                pts_cam_filtered = pts_cam_filtered[idx]
                
            views.append({
                "name": s_dir.name,
                "T_base_tcp": T_base_tcp,
                "pts_cam": pts_cam_filtered,
                "color_path": s_dir / "color.png"
            })
            
        print(f"Loaded {len(views)} views with segmented desk points.")
        
        if len(views) < 3:
            print("Not enough views with segmented desk points. Skipping.")
            continue
            
        # 3. Optimize T_TCP2CAM
        def unpack_tcp2cam(x):
            R = rotation_vector_to_matrix(x[0:3])
            t = x[3:6]
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            return T

        def objective(x):
            T_tcp2cam = unpack_tcp2cam(x)
            all_pts_base = []
            for v in views:
                T_base_cam = v["T_base_tcp"] @ T_tcp2cam
                pts_cam_h = np.hstack([v["pts_cam"], np.ones((len(v["pts_cam"]), 1))])
                pts_base = (T_base_cam @ pts_cam_h.T).T[:, :3]
                all_pts_base.append(pts_base)
            all_pts_base = np.vstack(all_pts_base)
            centroid = np.mean(all_pts_base, axis=0)
            centered = all_pts_base - centroid
            cov = centered.T @ centered
            evals, evecs = np.linalg.eigh(cov)
            normal = evecs[:, 0]
            dists = centered @ normal
            var = np.mean(dists**2)
            z_err = (centroid[2] - 0.7800)**2
            return var + z_err * 0.1

        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1500})
        T_TCP2CAM_opt = unpack_tcp2cam(res.x)
        T_CAM2TCP_opt = np.linalg.inv(T_TCP2CAM_opt)
        print(f"Optimization Success: {res.success}")
        print("Optimized T_CAM2TCP:")
        print(T_CAM2TCP_opt)
        
        # 4. Estimate board pose in base frame using multi-view consensus
        all_T_base_board = []
        
        for idx_v, v in enumerate(views):
            img = cv2.imread(str(v["color_path"]))
            if img is None:
                continue
            
            T_base_cam = v["T_base_tcp"] @ T_TCP2CAM_opt
            T_cam_base = np.linalg.inv(T_base_cam)
            
            # Segment desk plane in base frame
            pts_cam_h = np.hstack([v["pts_cam"], np.ones((len(v["pts_cam"]), 1))])
            pts_base = (T_base_cam @ pts_cam_h.T).T[:, :3]
            centroid = np.mean(pts_base, axis=0)
            centered = pts_base - centroid
            cov = centered.T @ centered
            evals, evecs = np.linalg.eigh(cov)
            normal_base = evecs[:, 0]
            if normal_base[2] < 0:
                normal_base = -normal_base
            d_base = -np.dot(normal_base, centroid)
            
            # Desk plane in camera coordinates
            n_cam = T_cam_base[:3, :3] @ normal_base
            d_cam = d_base + np.dot(normal_base, T_base_cam[:3, 3])
            
            z_axis = n_cam / np.linalg.norm(n_cam)
            x_axis = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(x_axis, z_axis)) > 0.9:
                x_axis = np.array([0.0, 1.0, 0.0])
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            R_plane = np.column_stack([x_axis, y_axis, z_axis])
            t_plane = -d_cam * z_axis
            
            # Run 3-DOF board estimator
            bgr = cv2.resize(img, (IW, IH))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            
            # Bounding mask to board region
            mask = np.zeros((IH, IW), dtype=np.uint8)
            T_cam_board_prior = T_cam_base @ T_base_board_gt
            pts_prior_720 = project(CORNER_3D, T_cam_board_prior[:3, :3], T_cam_board_prior[:3, 3])
            contour_720 = np.array([pts_prior_720[0], pts_prior_720[1], pts_prior_720[3], pts_prior_720[2]], dtype=np.int32)
            cv2.fillPoly(mask, [contour_720], 255)
            
            blobs, iscyl = detect_blobs(gray, mask)
            if len(blobs) < 4:
                continue
                
            params, cost, rv, tv = estimate(blobs, iscyl, R_plane, t_plane, n_th=36, n_uv=11, uv=0.15)
            
            T_cam_board = np.eye(4)
            T_cam_board[:3, :3] = rotation_vector_to_matrix(rv)
            T_cam_board[:3, 3] = tv.ravel()
            
            T_base_board_est = T_base_cam @ T_cam_board
            all_T_base_board.append(T_base_board_est)
            
        print(f"Obtained {len(all_T_base_board)} valid board pose estimates.")
        
        if len(all_T_base_board) == 0:
            print("Failed to estimate board pose. Skipping.")
            continue
            
        # Consensus board pose
        all_t = np.array([T[:3, 3] for T in all_T_base_board])
        t_consensus = np.median(all_t, axis=0)
        
        all_R = np.array([T[:3, :3] for T in all_T_base_board])
        R_sum = np.sum(all_R, axis=0)
        U, _, Vt = np.linalg.svd(R_sum)
        R_consensus = U @ Vt
        if np.linalg.det(R_consensus) < 0:
            Vt[2, :] *= -1
            R_consensus = U @ Vt
            
        T_base_board_consensus = np.eye(4)
        T_base_board_consensus[:3, :3] = R_consensus
        T_base_board_consensus[:3, 3] = t_consensus
        
        # 5. Project consensus board corners and save verification image
        sample_view = subdirs[0]
        color_path = sample_view / "color.png"
        pose_path = sample_view / "pose.txt"
        T_base_tcp = parse_pose_txt(pose_path)
        
        T_base_cam = T_base_tcp @ T_TCP2CAM_opt
        T_cam_base = np.linalg.inv(T_base_cam)
        T_cam_board = T_cam_base @ T_base_board_consensus
        
        pts_board_h = np.hstack([CORNER_3D, np.ones((4, 1))])
        pts_cam_proj = (T_cam_board @ pts_board_h.T).T[:, :3]
        
        pts_2d_h = pts_cam_proj @ K_full.T
        pts_2d = pts_2d_h[:, :2] / pts_2d_h[:, 2:3]
        
        img = cv2.imread(str(color_path))
        if img is not None:
            img_draw = cv2.resize(img, (960, 540))
            scale = 960.0 / 3840.0
            pts_2d_scaled = pts_2d * scale
            
            edges = [(0, 1), (1, 3), (3, 2), (2, 0)]
            for i, j in edges:
                pt1 = tuple(pts_2d_scaled[i].astype(int))
                pt2 = tuple(pts_2d_scaled[j].astype(int))
                cv2.line(img_draw, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
                
            out_name = f"verify_{p_folder.name}_depth_refined.png"
            cv2.imwrite(str(out_dir / out_name), img_draw)
            print(f"Saved refined verification image to out/{out_name}")
            
        # 6. Save results to dictionary
        # Compute desk plane params
        all_pts_base = []
        for v in views:
            T_base_cam = v["T_base_tcp"] @ T_TCP2CAM_opt
            pts_cam_h = np.hstack([v["pts_cam"], np.ones((len(v["pts_cam"]), 1))])
            pts_base = (T_base_cam @ pts_cam_h.T).T[:, :3]
            all_pts_base.append(pts_base)
        all_pts_base = np.vstack(all_pts_base)
        centroid = np.mean(all_pts_base, axis=0)
        centered = all_pts_base - centroid
        cov = centered.T @ centered
        evals, evecs = np.linalg.eigh(cov)
        normal_base = evecs[:, 0]
        if normal_base[2] < 0:
            normal_base = -normal_base
        d_base = -np.dot(normal_base, centroid)
        
        pose_results = {
            "T_CAM2TCP_opt": T_CAM2TCP_opt.tolist(),
            "desk_plane": {
                "normal": normal_base.tolist(),
                "d": float(d_base),
                "z_height_m": float(-d_base / normal_base[2])
            },
            "T_base_board_consensus": T_base_board_consensus.tolist()
        }
        
        results_json_path = out_dir / f"results_{p_folder.name}.json"
        with open(results_json_path, 'w') as f:
            json.dump(pose_results, f, indent=4)
        print(f"Saved Pose results to {results_json_path}")

if __name__ == "__main__":
    main()
