import json
import re
import cv2
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# 4K Camera Intrinsics
K_full = np.array([
    [2251.950927734375, 0.0, 1931.4296875],
    [0.0, 2251.7568359375, 1091.6837158203125],
    [0.0, 0.0, 1.0]
])

# Board 3D corners in board frame
CORNER_3D = np.float32([
    [-0.16, 0.125, 0.025],   # Top-Left
    [ 0.16, 0.125, 0.025],   # Top-Right
    [-0.16, -0.125, 0.025],  # Bottom-Left
    [ 0.16, -0.125, 0.025]   # Bottom-Right
])

# Pointer tip relative to end effector
V_TIP = np.array([-0.0043625, 0.00081629, 0.34239859])

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
    
    # Quaternion to rotation matrix
    x, y, z, w = rx, ry, rz, rw
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
    ])
    T[:3, :3] = R
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

def main():
    json_path = ROOT / "glunet_pose_refinement2/out/gtsam_optimized_poses.json"
    with open(json_path, 'r') as f:
        opt_data = json.load(f)
        
    T_EE2cam_opt = np.array(opt_data["T_EE2cam_opt"])
    
    # Standard edges for drawing
    edges = [(0, 1), (1, 3), (3, 2), (2, 0)]
    
    for p_idx in [1, 2, 3]:
        p_folder = ROOT / "images/images_depth" / f"pose_{p_idx}"
        
        # Load touch-probe corners in base frame
        gt_path = p_folder / "ground_truth.txt"
        c_data = parse_ground_truth_corners(gt_path)
        ordered = get_corners_in_order(c_data)
        pts_base_gt = np.array([R @ V_TIP + t for R, t in ordered])
        T_base_board_gt = kabsch_alignment(CORNER_3D, pts_base_gt)
        
        # Load optimized board pose
        T_base_board_opt = np.array(opt_data["poses"][f"pose_{p_idx}"]["T_base_board_opt"])
        
        # Let's find one view to draw
        subdirs = sorted([d for d in p_folder.iterdir() if d.is_dir()])
        s_dir = subdirs[0] # Just draw the first view
        
        T_base_tcp = parse_pose_txt(s_dir / "pose.txt")
        T_base_cam = T_base_tcp @ T_EE2cam_opt
        T_cam_base = np.linalg.inv(T_base_cam)
        
        # Project GT board corners to camera
        pts_gt_cam_proj = (T_cam_base @ T_base_board_gt @ np.hstack([CORNER_3D, np.ones((4, 1))]).T).T[:, :3]
        pts_gt_2d_h = pts_gt_cam_proj @ K_full.T
        pts_gt_2d = pts_gt_2d_h[:, :2] / pts_gt_2d_h[:, 2:3]
        
        # Project Optimized board corners to camera
        pts_opt_cam_proj = (T_cam_base @ T_base_board_opt @ np.hstack([CORNER_3D, np.ones((4, 1))]).T).T[:, :3]
        pts_opt_2d_h = pts_opt_cam_proj @ K_full.T
        pts_opt_2d = pts_opt_2d_h[:, :2] / pts_opt_2d_h[:, 2:3]
        
        # Draw on image
        img_path = s_dir / "color.png"
        img = cv2.imread(str(img_path))
        if img is not None:
            img_draw = cv2.resize(img, (960, 540))
            scale = 960.0 / 3840.0
            
            pts_gt_scaled = pts_gt_2d * scale
            pts_opt_scaled = pts_opt_2d * scale
            
            # Draw GT (Red)
            for i, j in edges:
                pt1 = tuple(pts_gt_scaled[i].astype(int))
                pt2 = tuple(pts_gt_scaled[j].astype(int))
                cv2.line(img_draw, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
                
            # Draw Optimized (Green)
            for i, j in edges:
                pt1 = tuple(pts_opt_scaled[i].astype(int))
                pt2 = tuple(pts_opt_scaled[j].astype(int))
                cv2.line(img_draw, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
                
            # Label
            cv2.putText(img_draw, f"Pose {p_idx} (Red: GT touch-probe, Green: Optimized Board)", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            out_path = ROOT / f"glunet_pose_refinement2/out/gtsam_verify_pose_{p_idx}.png"
            cv2.imwrite(str(out_path), img_draw)
            print(f"Saved visualization: {out_path}")

if __name__ == '__main__':
    main()
