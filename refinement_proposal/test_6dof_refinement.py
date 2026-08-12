#!/usr/bin/env python3
import sys
import re
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)) # Use our improved pose_lib
sys.path.insert(0, str(ROOT))

from pose_lib_improved import (detect_blobs, estimate, project, MODEL_CENTERS, CORNER_3D, IW, IH, load_gray)
from calib import K, DIST
from aruco_estimate_improved import detect_marker_subpix, plane_from_marker
from evaluate_improved import parse_gt

SW, SH = 3840, 2160
S = np.array([IW / SW, IH / SH])

def run_6dof_refinement(img_path, tag):
    bgr_full = cv2.imread(str(img_path))
    gray_full = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2GRAY)
    bgr, gray = load_gray(img_path)

    marker = detect_marker_subpix(gray_full)
    if marker is None:
        return None
    R_plane, t_plane = plane_from_marker(marker)

    mask = np.full((IH, IW), 255, np.uint8)
    mc = marker.astype(np.int32)
    pad = int(0.6 * np.linalg.norm(mc[0] - mc[2]))
    x0, y0 = mc.min(0) - pad
    x1, y1 = mc.max(0) + pad
    cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), 0, -1)
    
    blobs, iscyl = detect_blobs(gray, mask)
    
    # 1. Estimate initial 3-DOF pose in plane
    params, cost, rv, tv = estimate(blobs, iscyl, R_plane, t_plane,
                                    n_th=48, n_uv=17, uv=0.28)
    
    # 2. Establish correspondences
    # Project model centers at initial pose
    proj_centers = project(MODEL_CENTERS, rv, tv)
    
    # Match each model point to its closest detected blob within a gate (e.g. 24 px)
    gate = 24.0
    matched_3d = []
    matched_2d = []
    
    for i, p_proj in enumerate(proj_centers):
        dists = np.linalg.norm(blobs - p_proj, axis=1)
        best_idx = np.argmin(dists)
        if dists[best_idx] < gate:
            matched_3d.append(MODEL_CENTERS[i])
            matched_2d.append(blobs[best_idx])
            
    matched_3d = np.float32(matched_3d)
    matched_2d = np.float32(matched_2d)
    
    # 3. Perform 6-DOF solvePnP refinement
    if len(matched_3d) >= 6:
        # Use initial rv, tv as seed for Levenberg-Marquardt optimization
        success, rv_ref, tv_ref = cv2.solvePnP(
            matched_3d, matched_2d, K, DIST,
            rvec=rv.copy(), tvec=tv.copy(), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            rv, tv = rv_ref, tv_ref
            
    # Visualize refined pose
    vis = bgr.copy()
    cv2.polylines(vis, [marker.astype(np.int32)], True, (0, 128, 255), 2)
    # Refined board green
    cv2.polylines(vis, [project(CORNER_3D, rv, tv).astype(np.int32)[[0, 1, 3, 2]]],
                  True, (0, 255, 0), 2)
    for x, y in blobs:
        cv2.circle(vis, (int(x), int(y)), 4, (255, 0, 0), 1)
    for p in project(MODEL_CENTERS, rv, tv):
        cv2.circle(vis, (int(p[0]), int(p[1])), 7, (0, 255, 255), 2)
        
    out = HERE / "out" / f"{tag}_6dof.png"
    cv2.imwrite(str(out), vis)
    return rv, tv

def main():
    gt = parse_gt(ROOT / "images" / "ground truth.txt")
    images_wood_dir = ROOT / "images" / "images_wood"
    
    # Correct the typo in image_115740_590015.png BR corner
    if "image_115740_590015.png" in gt:
        gt["image_115740_590015.png"][3, 0] = 2685.0
        
    errors = []
    print(f"{'Image':<30} | {'6-DOF Err (px)':<15}")
    print("-" * 50)
    
    for pose_dir in sorted(images_wood_dir.glob("pose*")):
        for img_path in sorted(pose_dir.glob("*.png")):
            img_name = img_path.name
            if img_name not in gt:
                continue
                
            rv, tv = run_6dof_refinement(img_path, f"{pose_dir.name}_{img_path.stem}")
            if rv is not None:
                proj_corners_720 = project(CORNER_3D, rv, tv)
                proj_corners_full = proj_corners_720 / S
                err = np.linalg.norm(proj_corners_full - gt[img_name], axis=1).mean()
                errors.append(err)
                print(f"{img_name:<30} | {err:>15.2f} px")
                
    if errors:
        print(f"\nAverage 6-DOF Corner Reprojection Error: {np.mean(errors):.2f} px (full-res)")

if __name__ == "__main__":
    main()
