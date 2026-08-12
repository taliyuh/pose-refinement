#!/usr/bin/env python3
import sys
import re
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)) # Use our improved pose_lib
sys.path.insert(0, str(ROOT))

from pose_lib_improved import (detect_blobs, estimate, project, CORNER_3D, IW, IH, load_gray)
from calib import K, DIST
from aruco_estimate_improved import detect_marker_subpix, plane_from_marker

def parse_gt(gt_path):
    gt_dict = {}
    content = Path(gt_path).read_text()
    # Normalize zero-width spaces and other non-ascii stuff
    content = re.sub(r'[^\x00-\x7F]+', '', content)
    
    # Simple regex parsing
    pattern = r'(image_\d+_\d+\.png):\s*([\d\s,;]+)'
    matches = re.findall(pattern, content)
    for name, pts_str in matches:
        pts = []
        for p in pts_str.split(';'):
            p = p.strip()
            if not p:
                continue
            coords = list(map(float, p.split(',')))
            pts.append(coords)
        if len(pts) == 4:
            gt_dict[name] = np.array(pts, dtype=np.float32)
            
    # Also handle the two images at the top of the file:
    top_pattern = r'(image_\d+_\d+\.png):\s*top left:\s*(\d+),\s*(\d+)\s*top right:\s*(\d+),\s*(\d+)\s*bottom left:\s*(\d+),\s*(\d+)\s*bottom right:\s*(\d+),\s*(\d+)'
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

SW, SH = 3840, 2160
S = np.array([IW / SW, IH / SH])

def main():
    gt = parse_gt(ROOT / "images" / "ground truth.txt")
    images_wood_dir = ROOT / "images" / "images_wood"
    
    # Correct the typo in image_115740_590015.png BR corner: change 1685 to 2685
    if "image_115740_590015.png" in gt:
        gt["image_115740_590015.png"][3, 0] = 2685.0
        
    errors = []
    
    print(f"{'Image':<30} | {'Est Err (px)':<12} | {'Cost (px)':<10} | {'Yaw (deg)':<10}")
    print("-" * 70)
    
    for pose_dir in sorted(images_wood_dir.glob("pose*")):
        for img_path in sorted(pose_dir.glob("*.png")):
            img_name = img_path.name
            if img_name not in gt:
                continue
                
            bgr_full = cv2.imread(str(img_path))
            gray_full = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2GRAY)
            bgr, gray = load_gray(img_path)

            marker = detect_marker_subpix(gray_full)
            if marker is None:
                continue
                
            R_plane, t_plane = plane_from_marker(marker)

            # mask out marker
            mask = np.full((IH, IW), 255, np.uint8)
            mc = marker.astype(np.int32)
            pad = int(0.6 * np.linalg.norm(mc[0] - mc[2]))
            x0, y0 = mc.min(0) - pad
            x1, y1 = mc.max(0) + pad
            cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), 0, -1)
            
            blobs, iscyl = detect_blobs(gray, mask)
            params, cost, rv, tv = estimate(blobs, iscyl, R_plane, t_plane,
                                            n_th=48, n_uv=17, uv=0.28)
            
            # Project corners
            proj_corners_720 = project(CORNER_3D, rv, tv)
            proj_corners_full = proj_corners_720 / S
            
            gt_corners = gt[img_name]
            err = np.linalg.norm(proj_corners_full - gt_corners, axis=1).mean()
            errors.append(err)
            
            print(f"{img_name:<30} | {err:>12.2f} | {cost:>10.2f} | {np.degrees(params[0]):>10.1f}")
            
    if errors:
        print(f"\nAverage Mean Corner Reprojection Error: {np.mean(errors):.2f} px (full-res)")
    else:
        print("\nNo annotated images evaluated.")

if __name__ == "__main__":
    main()
