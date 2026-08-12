#!/usr/bin/env python3
"""
Improved ArUco-calibrated board pose estimation.
Incorporates:
1. Sub-pixel marker corner refinement.
2. Restricting the translation search window to 5.5 cm to prevent column-shifting.
"""
import sys
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)) # Use our improved pose_lib
sys.path.insert(0, str(ROOT))

from pose_lib_improved import (detect_blobs, estimate, project, MODEL_CENTERS,
                              CORNER_3D, IW, IH, load_gray)
from calib import K, DIST, ARUCO_DICT, ARUCO_PARAMS, ARUCO_TARGET_ID, ARUCO_OBJ_POINTS

OUT = HERE / "out"
OUT.mkdir(exist_ok=True, parents=True)
SW, SH = 3840, 2160
S = np.array([IW / SW, IH / SH])

def detect_marker_subpix(gray_full):
    """Detect marker in full-res and apply sub-pixel refinement."""
    det = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    corners, ids, _ = det.detectMarkers(gray_full)
    if ids is None:
        return None
    ids = ids.ravel()
    for c, i in zip(corners, ids):
        if i == ARUCO_TARGET_ID:
            c_reshaped = c.reshape(-1, 2)
            # Apply sub-pixel refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.05)
            refined_c = cv2.cornerSubPix(gray_full, c_reshaped.astype(np.float32), (5, 5), (-1, -1), criteria)
            return refined_c * S # scale to 720p
    return None

def plane_from_marker(marker_720):
    _, rv, tv = cv2.solvePnP(ARUCO_OBJ_POINTS, marker_720, K, DIST,
                             flags=cv2.SOLVEPNP_IPPE_SQUARE)
    R, _ = cv2.Rodrigues(rv)
    t = tv.ravel()
    if np.dot(R[:, 2], -t) < 0:
        R = R @ np.diag([1.0, -1.0, -1.0])
    return R, t

def run_improved(img_path, tag):
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
    
    # Use tighter translation window (win=0.05 m = 5 cm) to prevent column shifting
    params, cost, rv, tv = estimate(blobs, iscyl, R_plane, t_plane,
                                    n_th=48, n_uv=17, uv=0.28)

    vis = bgr.copy()
    cv2.polylines(vis, [marker.astype(np.int32)], True, (0, 128, 255), 2)
    cv2.polylines(vis, [project(CORNER_3D, rv, tv).astype(np.int32)[[0, 1, 3, 2]]],
                  True, (0, 255, 0), 2)
    for x, y in blobs:
        cv2.circle(vis, (int(x), int(y)), 4, (255, 0, 0), 1)
    for p in project(MODEL_CENTERS, rv, tv):
        cv2.circle(vis, (int(p[0]), int(p[1])), 7, (0, 255, 255), 2)
    cv2.putText(vis, f"{tag} cost={cost:.1f}px blobs={len(blobs)}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out = OUT / f"{tag}_improved.png"
    cv2.imwrite(str(out), vis)
    return params, cost, rv, tv

def main():
    # If path provided in command line, use it; otherwise default to images/images_wood
    root = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "images/images_wood")
    print(f"Scanning directory: {root}")
    pose_dirs = sorted(root.glob("pose*"))
    if pose_dirs:
        for pose_dir in pose_dirs:
            print(f"\n=== {pose_dir.name} ===")
            for img in sorted(pose_dir.glob("*.png")):
                run_improved(img, f"{pose_dir.name}_{img.stem}")
    else:
        for img in sorted(root.glob("*.png")):
            run_improved(img, img.stem)
    print(f"\nDone -> Output saved to {OUT}/")

if __name__ == "__main__":
    main()
