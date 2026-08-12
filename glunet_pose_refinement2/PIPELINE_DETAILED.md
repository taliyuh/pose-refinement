# GLU-Net Pose Refinement v2: Detailed Pipeline Walkthrough

This document explains, step by step, how the `glunet_pose_refinement2` pipeline works in code.

It covers:
- image preprocessing
- ArUco-based coarse geometry
- depth-based desk-plane estimation
- object/blob detection and matching against the board model
- dense GLU-Net correspondence estimation
- corner transfer and pose recovery
- multi-view depth refinement
- global GTSAM optimization
- outputs and debugging paths

---

## 1. What the pipeline is solving

The core task is to estimate the 6DoF board pose in camera coordinates for depth-camera captures.

The system does not rely on only one cue. Instead, it combines:
1. **Geometric priors** from ArUco marker pose and desk plane.
2. **Board-structure cues** from bright cutout/blob detections (shape centers).
3. **Dense photometric correspondences** from GLU-Net optical flow.
4. **Cross-view consistency** through multi-view depth refinement and factor-graph optimization.

Main scripts and their role:
- `estimate_glunet.py`: single-pair pipeline and all-pairs evaluation.
- `run_refinement_all.py`: batch pairing over `images/images_depth/pose_*`.
- `run_depth_refinement.py`: depth-plane + hand-eye refinement + board consensus pose per pose folder.
- `solve_poses_gtsam.py`: global optimization over hand-eye, camera poses, marker poses, and board poses.
- `visualize_results.py`: projects optimized board and compares to touch-probe GT.

Shared geometry and detection internals come from:
- `refinement_proposal/pose_lib_improved.py`
- `refinement_proposal/aruco_estimate_improved.py`

---

## 2. Coordinate systems and core constants

### 2.1 Image scales

There are two image resolutions used repeatedly:
- Full-resolution depth camera images: **3840x2160**.
- Working resolution for blob/cost optimization: **1280x720**.

Scale vector in `estimate_glunet.py` and pose libs:
- `S = [IW/SW, IH/SH] = [1280/3840, 720/2160]`.

Convention:
- Full-res to 720p: multiply by `S`.
- 720p to full-res: divide by `S`.

### 2.2 Board model geometry

From `pose_lib_improved.py`:
- `CORNER_3D`: 4 board corners (meters), at top face `z=0.025`.
- `MODEL_CENTERS`: 16 CAD-derived shape-center points (4 columns x 4 rows; circle/hex/triangle/rect columns).
- `CYL_IDX = [0,1,2,3]`: first column used as circular/cylinder anchors.

These 3D points are what the optimizer projects into 2D and aligns to detected blobs.

### 2.3 Camera intrinsics

Two intrinsic sets appear:
- 720p intrinsics (from `calib.py` and `pose_lib_improved.py`) used by `solvePnP` and projection during 720p processing.
- 4K intrinsics (`K_new` in several scripts) used for undistortion and full-res projection.

Distortion handling:
- For depth-camera captures, `load_image` in `estimate_glunet.py` applies `cv2.undistort` with `K_new`, `D_new`.
- Ground-truth corners are also undistorted before comparison in `undistort_corners(...)`.

---

## 3. Stage A - Coarse board pose from ArUco + blob geometry

This stage is implemented by:
- `estimate_corners_from_aruco(...)`
- `estimate_corners_from_aruco_img(...)`
- helpers from `aruco_estimate_improved.py` and `pose_lib_improved.py`

### 3.1 Marker detection with robust fallback

Function: `detect_marker_subpix_robust(gray_full)`

Procedure:
1. Configure ArUco detector with tuned thresholds and subpixel refinement.
2. Detect on raw grayscale.
3. If fail, detect on CLAHE-enhanced grayscale.
4. For the target marker id (`ARUCO_TARGET_ID`), refine corners via `cv2.cornerSubPix`.
5. Return corners in 720p coordinates (scaled by `S`).

Why this matters:
- Marker corners anchor plane orientation robustly before any learned matching.

### 3.2 Plane from marker

Function: `plane_from_marker(marker_720)` in `aruco_estimate_improved.py`.

Procedure:
1. `cv2.solvePnP(..., SOLVEPNP_IPPE_SQUARE)` on marker object corners.
2. Convert `rvec` to `R`.
3. Enforce normal orientation consistency (flip if needed).

Output:
- `R_plane`, `t_plane` prior for board-in-plane optimization.

### 3.3 Optional depth-plane override

If depth is available, `get_plane_from_depth(img_path)` is used.

Procedure:
1. Load sibling `depth.png`.
2. Keep valid depth interval (300..1000) mm.
3. Convert depth to meters and multiply by pose-folder scale correction:
   - pose_1: 0.9510
   - pose_2: 0.9503
   - pose_3: 0.9502
4. Backproject sampled pixels into camera-space 3D using full-res intrinsics.
5. Downsample to <=3000 points.
6. RANSAC plane fit (200 iters, 3-point model, 3 mm threshold).
7. Refit normal with SVD on inliers.
8. Force normal orientation (`n_z < 0` in this implementation).

Then in `estimate_corners_from_aruco_img(...)`:
- Rotate marker normal to depth normal (Rodrigues-style alignment via cross product/skew matrix).
- Project marker translation onto the depth plane.

Effect:
- Keeps orientation from marker but corrects plane placement with measured depth surface.

### 3.4 ROI masking and marker exclusion

Before blob detection, a board-expected ROI is built from marker geometry:
- a rectangle around the marker center expanded asymmetrically (board expected mostly on one side)
- marker area is zeroed out with padding to avoid false blobs from marker graphics

### 3.5 Blob detection internals

Function: `detect_blobs(gray, roi, element_size=35, thr=60)` in `pose_lib_improved.py`.

Exact steps:
1. Top-hat morphology with elliptical SE (default 35x35).
2. Min-max normalization to 8-bit.
3. Binary threshold at 60.
4. Apply ROI mask.
5. Morphological open (3x3) then close (5x5).
6. Find external contours.
7. Filter contours by:
   - area in [40, 2500]
   - solidity-like ratio `area / area(convex hull) >= 0.75`
8. Compute centroid from moments.
9. Compute circularity: `4*pi*A/P^2`.
10. Classify as cylinder-like if circularity > 0.78.

Returns:
- `blobs`: Nx2 centroids.
- `iscyl`: boolean flags.

### 3.6 3-DOF board pose optimization on desk plane

Function: `estimate(blobs, iscyl, R_plane, t_plane, n_th, n_uv, uv)`.

Model assumptions:
- Board motion constrained to a plane; pose parameterized by
  - yaw `theta`
  - in-plane translations `(u,v)`
- `pose_in_plane(...)` composes `R = R_plane * Rz(theta)` and `t = t_plane + R_plane*[u,v,0]`.

Cost design (`cost_fn`):
1. Project CAD model centers (cylinder subset + other subset) into image.
2. Reject if not enough projected points in bounds.
3. Compute assignment cost with Hungarian algorithm (`linear_sum_assignment`) under gate distance.
4. Penalize unmatched model points by gate value.
5. Weighted blend:
   - cylinder group weighted by `cyl_w` (default 2.0)
   - others weighted by 1

Optimization schedule:
1. Coarse gate: 48 px.
2. Fine gate: 22 px.
3. Coarse yaw + translation grid search (`n_th`, `n_uv`).
4. Additional seed from `cylinder_pose_seed(...)` using collinear cylinder detection to resolve 180 deg ambiguity.
5. Local grid refinement around each seed.
6. Nelder-Mead continuous refinement.
7. Select global best candidate.

Outputs:
- best `[theta,u,v]`
- scalar cost
- `rvec`, `tvec`

### 3.7 Coarse corners

Finally:
- `project(CORNER_3D, rv, tv)` gives 720p corners.
- divide by `S` to return full-res corners.

These coarse corners are later used to crop/rectify images for GLU-Net.

---

## 4. Stage B - Image preprocessing for GLU-Net

### 4.1 Why preprocessing exists

Wood texture and illumination gradients can dominate raw RGB matching. The code intentionally enhances only small bright structures associated with board features.

### 4.2 Top-hat preprocessing

Function: `apply_tophat_prep(img, se_sz=25)` in `estimate_glunet.py`.

Steps:
1. Convert BGR to grayscale.
2. Morphological top-hat with elliptical 25x25 element.
3. Normalize to [0,255].
4. Replicate single channel back to 3 channels.

This representation is fed to GLU-Net for both source and target.

---

## 5. Stage C - Dense correspondence estimation (GLU-Net)

### 5.1 Model loading

`load_glunet(device)` builds `GLU_Net` with pretrained bundle:
- model type: `DPED_CityScape_ADE`
- cyclic consistency enabled
- iterative refinement enabled
- consensus network disabled

### 5.2 Preferred path: board rectification before flow

Function: `dense_matches(...)` in `estimate_glunet.py`.

When coarse source and target corners are available:
1. Define canonical board plane image coordinates:
   - TL(0,0), TR(512,0), BL(0,400), BR(512,400)
2. Estimate `H_rect_src`, `H_rect_tgt` with `cv2.findHomography`.
3. Warp both images to canonical board views (`cv2.warpPerspective`).
4. Run GLU-Net flow from target rectified to source rectified.
5. Convert dense grid to correspondence pairs:
   - `pts_t`: target pixels in rectified grid
   - `pts_s`: target pixels displaced by flow
6. Fit rectified homography to detect 180 deg flip:
   - if `H[0,0] + H[1,1] < 0`, reverse target-corner order and re-run.
7. Map rectified correspondences back to full image with inverse homographies.

### 5.3 Fallback path: direct resized-image flow

If board rectification is not possible:
1. Resize source/target to max dimension (default 1024).
2. Apply top-hat preprocessing.
3. Run GLU-Net flow directly.
4. Rescale coordinates back to original image sizes.

Output always:
- `pts_source`, `pts_target` dense match arrays.

---

## 6. Stage D - Match filtering and corner transfer

Implemented mainly in `evaluate_pair(...)`.

### 6.1 Source-board spatial filtering

1. Build polygon mask from source board corners.
2. Keep matches whose source endpoint lies inside polygon.
3. If source pose exists, project 16 model centers and carve circular holes in mask to reject cutout interior regions (prefer flat top-surface correspondences).

Radius logic:
- `r_px = round(0.022 * 2252 / tz)` where `tz` from source pose.

### 6.2 Robust homography estimation

`cv2.findHomography(..., cv2.USAC_MAGSAC, 2.0)` on filtered matches.

If low inlier count (`<10000`), try explicit 180 deg target-corner flip and keep the variant with more inliers.

### 6.3 Corner prediction and pose recovery

1. Transform source corners by homography to get predicted target corners.
2. Convert predicted full-res corners to 720p (`*S`).
3. Solve PnP using `solve_corners(...)`.
4. Reproject board corners and 16 model centers from recovered pose.

Error metric (if GT exists):
- mean 2D corner distance in full-res pixels.

---

## 7. Stage E - Depth-assisted 3D handling

Depth enters in two places:

### 7.1 Plane estimation for coarse pose (already described)

`get_plane_from_depth(...)` improves plane normal and translation for the constrained estimator.

### 7.2 Backprojection utilities

`estimate_glunet.py` includes:
- `backproject_to_board_plane(...)`: ray-plane intersection to board `Z=0.025` when depth unavailable.
- `backproject_using_depth(...)`: uses per-pixel depth + camera intrinsics to recover 3D points then maps to board frame.

Depth behavior:
- If depth missing or zero, it falls back to plane backprojection.
- Uses the same per-folder scale correction factors.

These utilities support future/extended correspondence lifting workflows.

---

## 8. Stage F - Batch execution over depth folders

Script: `run_refinement_all.py`

Flow:
1. Load GT corner map from `images/ground truth.txt`.
2. Correct one known GT typo if present.
3. Load GLU-Net model once.
4. For each `images/images_depth/pose_*` folder:
   - choose first subfolder image as source
   - evaluate source against all remaining subfolders via `evaluate_pair(...)`
5. Save per-pair collage outputs in `glunet_pose_refinement2/out`.

---

## 9. Stage G - Multi-view depth refinement (`run_depth_refinement.py`)

This script is a stronger geometric refinement pipeline that does not only depend on pairwise homographies.

### 9.1 Inputs

For each `pose_*` folder:
- many views (`<timestamp>/color.png`, `<timestamp>/depth.png`, `<timestamp>/pose.txt`)
- touch-probe corner GT (`ground_truth.txt`) used as coordinate prior

### 9.2 Build GT board prior in base frame

1. Parse touch-probe transforms for 4 corners.
2. Compute measured corner points from end-effector tip offset `V_TIP`.
3. Align CAD `CORNER_3D` to measured points with Kabsch.
4. Obtain `T_base_board_gt`.

### 9.3 Per-view desk-plane extraction

For each depth image:
1. backproject depth to camera points
2. RANSAC plane segmentation
3. keep inlier cloud
4. store reduced cloud with associated robot pose `T_base_tcp`

### 9.4 Optimize hand-eye (`T_TCP2CAM`)

Unknowns are represented as 6 params (Rodrigues + translation), optimized with Nelder-Mead.

Objective:
1. transform all per-view camera desk points to base frame
2. stack all points
3. compute smallest-eigenvector plane normal
4. minimize:
   - variance of distances to plane
   - soft penalty on plane centroid `z` deviation from 0.7800 m

This yields optimized hand-eye transform.

### 9.5 Per-view board estimation under refined geometry

For each valid view:
1. transform base-plane normal into camera frame
2. build `R_plane`, `t_plane`
3. run blob detection restricted by projected board prior mask
4. run 3-DOF estimator (`estimate(...)`)
5. convert `T_cam_board` to `T_base_board`

### 9.6 Consensus board pose

Across all valid per-view board estimates:
- translation: median over views
- rotation: SVD projection of sum of rotation matrices to nearest valid rotation

Save:
- `results_pose_X.json`
- verification image overlay

---

## 10. Stage H - Global graph optimization (`solve_poses_gtsam.py`)

This script jointly optimizes hand-eye and board poses over all views/poses.

### 10.1 Variables

- `X`: hand-eye transform (`T_EE2cam`)
- `B_p`: board pose in base for each pose folder `p`
- `M_p`: marker pose in base for each pose folder `p`
- `C_i`: camera pose in base for each individual view

### 10.2 Measurements

For each view:
1. Kinematic relation from robot:
   - `C_i = T_base_tcp_i * X`
   - implemented via custom factor (`make_hand_eye_factor`)
2. ArUco observation (if marker detected):
   - Between factor on `(C_i, M_p)` with measured `T_cam_marker`
3. GLU-Net board observation (if board estimated):
   - Between factor on `(C_i, B_p)` with measured `T_cam_board`

### 10.3 Noise models

- Diagonal Gaussian terms tuned separately for kinematic, marker, and board factors.
- Marker and board factors wrapped in robust Huber noise to reduce outlier impact.

### 10.4 Optimization

- Initialize with fixed seed transform and first-available observations.
- Run Levenberg-Marquardt on the full factor graph.
- Extract optimized `T_EE2cam_opt`, `T_base_board_opt` for each pose folder.

### 10.5 Verification against touch-probe GT

For each pose folder:
1. build GT board transform via Kabsch from touch-probe corners.
2. compare optimized and GT transforms:
   - translation center error (mm)
   - geodesic rotation error (deg)
   - per-corner 3D error (mm)
3. save all metrics to `out/gtsam_optimized_poses.json`.

---

## 11. Visualization and debug artifacts

### 11.1 Pair-level collage (`evaluate_pair`)

Saved image contains 4 panes:
1. source with source corners
2. target with:
   - GT contour (red, if available)
   - GLU-Net pose contour (green)
   - ArUco contour (magenta, if available)
   - projected shape centers (yellow)
3. warped-source/target blend
4. sampled inlier match lines

### 11.2 Depth refinement verification

`run_depth_refinement.py` saves `verify_pose_X_depth_refined.png`.

### 11.3 GTSAM visualization

`visualize_results.py` projects:
- touch-probe GT board edges (red)
- optimized board edges (green)

per pose folder into one representative image.

---

## 12. Object finding and matching summary (the short version)

If you only care about "how objects are found and matched":

1. **Find candidate object blobs** using top-hat + threshold + morphology + contour filtering.
2. **Classify circle-like blobs** via circularity (used as strong orientation anchors).
3. **Project 16 known CAD centers** under candidate plane pose.
4. **Assign model points to observed blobs** with Hungarian matching under gate.
5. **Optimize yaw + in-plane translation** by minimizing gated assignment cost.
6. **Use cylinder-line seeding** to avoid 180 deg flips and bad local minima.
7. **Refine with Nelder-Mead** and keep global best.

This is the geometric backbone that gives robust coarse board corners before/alongside GLU-Net dense matching.

---

## 13. Typical end-to-end execution path

A common production-like flow is:
1. Run `run_refinement_all.py` to generate pairwise overlays and inspect quality.
2. Run `run_depth_refinement.py` for per-pose geometric consensus using depth.
3. Run `solve_poses_gtsam.py` for global fused optimization.
4. Run `visualize_results.py` to visually compare optimized vs touch-probe GT.

---

## 14. Failure modes and what each one usually means

1. ArUco not detected:
- likely lighting/contrast issue, marker blur, or severe viewpoint.
- robust detector does raw + CLAHE, but can still fail.

2. Too few blobs or unstable cost:
- ROI may be wrong (marker localization off) or threshold/top-hat not suitable for the frame.
- heavy texture/specular highlights can create false contours.

3. Homography fails after GLU-Net:
- correspondences are noisy or mostly off-board.
- board mask filtering may remove too many points.

4. 180-degree ambiguity in pair transfer:
- code already retries flipped target-corner ordering and chooses better inlier count.

5. Plane drift in depth refinement:
- depth scale correction mismatch or weak RANSAC inliers in that view.

6. GTSAM converges but errors remain high:
- poor initialization in some folders, inconsistent marker/board detections, or outlier-heavy measurements despite robust loss.

---

## 15. Important implementation details to remember

- Ground-truth corners for depth-camera images are undistorted before being used as reference.
- Most geometric optimization and blob logic runs at 1280x720 for speed and stability.
- Full-res projection is restored at the end for final visual overlays and pixel error reporting.
- Depth and ArUco are fused at the plane level: ArUco gives orientation prior, depth improves actual plane placement.
- The board is treated as a structured constellation, not a generic texture patch. This is why shape-center matching remains robust when texture is weak.

---

## 16. Script entry points

- Pair estimation / evaluation:
  - `python glunet_pose_refinement2/estimate_glunet.py --source <path> --target <path>`
  - `python glunet_pose_refinement2/estimate_glunet.py --evaluate-all`

- Batch depth-pose folder processing:
  - `python glunet_pose_refinement2/run_refinement_all.py`

- Depth-based multi-view refinement:
  - `python glunet_pose_refinement2/run_depth_refinement.py`

- Global factor-graph optimization:
  - `python glunet_pose_refinement2/solve_poses_gtsam.py`

- Optimized pose visualization:
  - `python glunet_pose_refinement2/visualize_results.py`

---

If you want, this can be extended with a second companion markdown containing only equations and pseudo-code (cost functions, plane fitting, and factor-graph residual definitions), but this file is already mapped directly to the current implementation.