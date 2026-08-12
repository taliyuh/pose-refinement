# High-Precision 6-DoF Board Pose Refinement: Team Presentation & Technical Progress Report

---

## Executive Summary

We have transitioned our robot-to-target pose estimation pipeline from a basic single-marker (ArUco) calibration approach to a **multi-feature 3D Board Model Optimization** pipeline. 

By replacing the small ArUco marker with our $320 \times 250\text{ mm}$ board model (~2,400 feature points, 2-level 3D pocket geometry) and optimizing the Hand-Eye transformation matrix ($X_{\text{tcp\_cam}}$), we achieved a **44.8% average reduction in 3D base-frame target scatter across Poses 1–3**, pulling median spatial error down from **~2.75 mm to ~1.51 mm** (reaching **0.96 mm** on Pose 2).

---

## 1. How the Pipeline Works Now

### 1.1 Why the Board Beats the ArUco Marker
* **Target Extent & Lever Arm:** The ArUco marker is small ($100 \times 100\text{ mm}$) with only 4 corners. A $1.0\text{ mm}$ corner detection error on a marker injects massive rotational bias ($0.38^\circ$) into Hand-Eye solvers, which amplifies into **5+ mm position error** at $1\text{ m}$ reach.
* **The Board Advantage:** Our calibration target spans $320 \times 250\text{ mm}$ with 40 distinct contours. Crucially, it features **two-level 3D geometry** (pocket floors at $z = 3\text{ mm}$ and top surface rims at $z = 25\text{ mm}$), creating a $22\text{ mm}$ vertical baseline that allows full 6-DoF optical depth and tilt triangulation without depth-plane collapse.

```
       ArUco Marker (Old)                   Board Model (New)
 ┌──────────────────────────┐     ┌──────────────────────────────────┐
 │ • Extent: 100x100 mm    │     │ • Extent: 320x250 mm             │
 │ • Features: 4 corners    │ vs  │ • Features: 40 contours, 2400 pts│
 │ • Accuracy: ~1.0 mm      │     │ • Accuracy: ~0.25 mm             │
 │ • Flat 2D (no 3D tilt)   │     │ • 2-Level 3D Baseline (22 mm)    │
 └──────────────────────────┘     └──────────────────────────────────┘
```

---

### 1.2 Step-by-Step Board Detection & Pose Solver Mechanics

#### Step A: Top-Hat Morphological Preprocessing
To eliminate wood grain textures and illumination gradients, [`apply_tophat_prep`](file:///var/home/bartek/board_pose_obtain/glunet_pose_refinement2/PIPELINE_DETAILED.md#L211) applies a morphological top-hat filter with a $25 \times 25$ elliptical structuring element. This isolates only the bright cutout contours and hole rims.

#### Step B: Feature & Blob Centroid Extraction
[`detect_blobs(...)`](file:///var/home/bartek/board_pose_obtain/refinement_proposal/pose_lib_improved.py) extracts external contours and filters candidates:
* **Area Gating:** $40 \le \text{Area} \le 2500$ px
* **Convexity Ratio:** $\frac{\text{Area}}{\text{Area(Convex Hull)}} \ge 0.75$
* **Anchor Classification:** Contours with circularity $>0.78$ are tagged as cylinder anchors.

#### Step C: 3-DoF In-Plane Coarse Alignment
[`estimate(...)`](file:///var/home/bartek/board_pose_obtain/refinement_proposal/pose_lib_improved.py) maps the CAD model's 16 shape centers (`MODEL_CENTERS`) onto detected image blobs:
* Solves in-plane yaw ($\theta$) and translation $(u, v)$ using the Hungarian algorithm (`linear_sum_assignment`).
* Uses collinear cylinder anchor triplets to resolve 180° board rotation ambiguity.
* Runs continuous Nelder-Mead optimization to refine coarse PnP.

#### Step D: Subpixel 6-DoF Rim Edge Search
[`fit_rims6.py`](file:///var/home/bartek/board_pose_obtain/glunet_pose_refinement2/fit_rims6.py#L15-L25) projects top rims ($z=25\text{ mm}$) and pocket floor rims ($z=3\text{ mm}$) into the raw 4K image and conducts 1D subpixel gradient edge searches along normals over **~50,000 points**. It frees all 6 degrees of freedom ($x, y, z, \text{roll}, \text{pitch}, \text{yaw}$) optically.

#### Step E: Multi-View Bundle Adjustment
[`bundle_rims.py`](file:///var/home/bartek/board_pose_obtain/glunet_pose_refinement2/bundle_rims.py#L9-L30) executes Zhang-style joint bundle adjustment across views for:
1. **9 Shared Camera Intrinsics:** $(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3)$
2. **$N$ Per-View Camera Poses:** $T_{\text{cam\_board}}$
*Note:* The robot arm is intentionally **excluded** from this bundle to prevent kinematic joint noise from corrupting visual edge accuracy.

---

## 2. Current Empirical Results (Poses 1–3)

We quantitatively benchmarked **Board Calibration (New)** against **ArUco/Camera Calibration (Old)** across Poses 1, 2, and 3 using [`compare_median_scatter.py`](file:///var/home/bartek/board_pose_obtain/glunet_pose_refinement2/results_board_vs_camera/compare_median_scatter.py):

| Pose Directory | Metric | Camera/Marker Calib (Old) | Board Calib (New) | Net Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Pose 1** | **Median 3D Scatter** | 3.180 mm | **2.063 mm** | **+35.1%** |
| | **Corner RMS vs Touch Probe GT** | 7.506 mm | **5.779 mm** | **+23.0%** |
| | **Median Angular Error** | 0.314° | **0.287°** | **+8.6%** |
| **Pose 2** | **Median 3D Scatter** | 2.803 mm | **0.961 mm** | **+65.7%** |
| | **Corner RMS vs Touch Probe GT** | 10.817 mm | **5.541 mm** | **+48.8%** |
| | **Median Angular Error** | 0.332° | **0.279°** | **+16.0%** |
| **Pose 3** | **Median 3D Scatter** | 2.262 mm | **1.525 mm** | **+32.6%** |
| | **Corner RMS vs Touch Probe GT** | 8.712 mm | **9.027 mm** | `-3.6%` |
| | **Median Angular Error** | 0.338° | **0.342°** | `-1.2%` |
| **AVERAGE** | **Median 3D Scatter** | **2.749 mm** | **1.516 mm** | **+44.8% Error Reduction** |

### Key Takeaway:
* Board calibration reduced average spatial scatter in the robot base frame from **2.75 mm down to 1.52 mm**.
* On Pose 2, sub-millimeter median scatter (**0.96 mm**) was achieved.

---

## 3. How We Improved It Further (Measured, Not Proposed)

Vision is ~0.1 mm in the camera frame. Everything above that is the robot. We
tried three kinematic compensation layers; **one works, and the other two are
deleted.** Recording which is which is the point of this section.

### 3.1 Joint Offsets — WORKS, and is deployed

* **The Problem:** encoder zero errors. Each joint reads a constant offset from
  its true angle, and the error propagates through the whole arm.
* **The Model ([`kinematic_calib.py`](file:///var/home/bartek/board_pose_obtain/glunet_pose_refinement2/kinematic_calib.py)):**
  $\theta_{\text{eff}} = \theta + \Delta\theta$, fitted jointly with the
  hand-eye $X$ and a free board pose per folder. Only $\Delta\theta_{2..5}$ are
  identifiable: $\Delta\theta_1$ is absorbed by the board pose, $\Delta\theta_6$
  by $X$.
* **Result:** board-corner residual **1.342 → 0.866 mm median** (p90 2.99 →
  1.45). Leave-one-folder-out: **11/11 folds improved**, 1.243 → 0.787 mm. The
  fitted offsets reproduce to ~0.03° across two independent bundles, which is
  the evidence they are a property of the arm rather than a fit artefact.

### 3.2 Gravity Deflection and Droop — DO NOT WORK, deleted

* Both were tried: `--flex` (hand-eye varying with the gravity direction in the
  TCP frame, 18 parameters) and **droop** (FK rotated about a shoulder axis by
  an angle proportional to reach, 1 parameter — the actual physics).
* **flex:** 11% for 18 parameters, and it moved $X$ by 22 mm.
  **droop:** 2.19 → 2.17 mm, residual-vs-reach correlation unchanged (+0.43 →
  +0.44), fitted rate 1.18 °/m — 1.8° of sag at 1.5 m, physically absurd.
  **Joint compliance from the reported effort:** 4 µm on top of the offsets for
  six parameters. Fitted alone it returns $C_4 = 0.80$ °/Nm, which over
  joint 4's 0.65 Nm span produces 0.52° — essentially the fitted encoder offset
  $\Delta\theta_4 = -0.63°$. It was impersonating the offset, not measuring
  stiffness.
* **Why:** these captures reach **upward, not outward**. $|t_{\text{base\_tcp}}|$
  is 1252–1522 mm but horizontal reach is only 232–711 mm, so the gravity moment
  about the shoulder barely varies across the dataset.
* **The decisive measurement:** after the joint offsets, the per-view residual
  correlates with **nothing** — horizontal reach, $|t|$, TCP height, all six
  joint angles, all six efforts, capture order and session time all come in
  below $|r| = 0.15$ over 147 views. There is no deflection-shaped signal left.

### 3.3 Rim Bias — WORKS, and was where the tilt error lived

* **The Problem:** the STL outline is not where the intensity edge is. Over 144
  views the signed rim residual is **−0.733 px** on the chamfered top rims and
  **+0.306 px** on the pocket floor rims — the same sign in **144 of 144 views**.
* **Why it dominated:** tilt is triangulated from the 22 mm baseline between the
  two contour levels, so the *differential* bias between them is a tilt error by
  construction. Its mean (1.25 px) is **0.68° of tilt bias**; its scatter
  (sd 0.333 px) is **0.181° of tilt** — against a measured per-view tilt error
  of 0.182°. It accounted for essentially all of it.
* **The Model ([`board_model.rim_bias`](file:///var/home/bartek/board_pose_obtain/glunet_pose_refinement2/board_model.py)):**
  a constant radial offset per contour level, in mm on the board, fitted inside
  the bundle alongside the intrinsics. **Two parameters.**

| | before | after |
| :--- | :--- | :--- |
| rim residual MAD | 0.684 px | **0.265 px** |
| tilt, per view | 0.183° (0.722 mm) | **0.141° (0.557 mm)** |
| chain residual | 0.865 mm | **0.762 mm** |
| leave-one-folder-out | 0.787 mm | **0.645 mm** |
| multi-view split-half | 0.314 mm | **0.208 mm** |

* **What we did NOT ship:** the view-geometry term `b1·s`. It lowers the pixel
  residual, and it is very nearly a per-view in-plane translation of the board —
  a pose parameter. Fitted it runs to 1.74–2.11 mm, an order of magnitude past
  the effect it models, for 0.006 px. Same failure mode as §3.2; rejected on the
  same grounds.

### 3.4 Configuration-Dependent Hand-Eye — DOES NOT WORK, rejected by placebo

* **The Hypothesis:** after the rim-bias fix, $\theta_6$ (tool roll) became the
  strongest predictor of the leftover residual, $r = +0.337$ against 0.15 for
  everything else. The camera hangs on a stick; rolling it swings the mass.
* **It passed the usual gate.** Leave-one-folder-out: 0.645 → 0.515 mm,
  **11/11 folders improved**. The gravity-direction variant did better still,
  0.406 mm, also 11/11.
* **Then it failed the placebo.** Refitting against a direction with **no
  gravitational meaning at all** (base $+x$) scores **0.408 mm, 11/11** —
  identical. In this dataset the camera always points at the board, so arm
  orientation lives on a narrow manifold and every "fixed direction in the TCP
  frame" is nearly the same regressor (canonical correlations 0.997, 0.986).
* **The lesson, which is the transferable part:** leave-one-folder-out shows a
  parameter *transfers*; it cannot show the parameter *means what its name
  says*. A view-dependent vision error transfers across folders exactly as well
  as a robot property does. **Always run the placebo.**
* **What is real:** ~0.24 mm of reproducible, orientation-dependent error that
  is not in the arm — fitted amplitude is 2 mm at the board on a 135 mm stick,
  and it grows without plateau as the constraint is loosened. It belongs in the
  vision model, not the kinematic chain.

### 3.5 Multi-View Board Pose — WORKS, and is the production estimator

* **The Problem:** one shot gives 0.87 mm in the base frame, and the chain, not
  the vision, is what is wrong.
* **The Model ([`fit_board_multiview.py`](file:///var/home/bartek/board_pose_obtain/glunet_pose_refinement2/fit_board_multiview.py)):**
  one shared $T_{\text{base\_board}}$ fitted against every view's subpixel rim
  edges at once, with a per-view chain perturbation $\xi_i$ under a Gaussian
  prior:
  $$T_{\text{cam\_board}}^{(i)} = X^{-1}\exp(-\xi_i)\,\text{FK}(\theta_i + \Delta\theta)^{-1}\,T_{\text{base\_board}}$$
* **The prior is the whole design.** Per view, vision is 0.13 mm and the chain
  is 0.87 mm — the chain is 7× noisier. Tying the views rigidly to one board
  pose asserts the opposite and is *worse* than plain pose averaging. The slack
  term is what makes the multi-view constraint pay.
* **Result:** split-half over the views, all 11 folders — **0.208 mm** at the
  board corner against 0.360 mm for pose averaging, and it wins in **11 of 11**
  folders. Worst folder 0.465 mm against 1.444 mm.

---

## 4. Next Implementation Milestones

1. **~~Model the view-dependent error as a symmetric rim term~~ — tried, does
   not work.** The symmetric terms (`b2·|s|`, `b3·|u_xy|`) are safe by
   construction — they change rim size, not position — and they improve the
   pixel residual (0.265 → 0.250 px). They make the pose *worse*: LOFO
   0.645 → 0.648 mm, multi-view split-half 0.208 → 0.221 mm, worse in 7 of 11
   folders. **This rules out a rim-geometry cause.** The ~0.24 mm behaves like a
   per-view pose perturbation correlated with camera orientation, so the next
   places to look are the camera intrinsics (a residual distortion error would
   do exactly this) and the arm — not the board model.
2. **~~Fix the tilt term~~ — done, see §3.3.** It was the differential rim bias
   between the two contour levels. Per-view tilt is now 0.141° (0.557 mm at the
   corner) from 0.183°, and 0.68° of constant tilt bias is gone. It is still the
   largest single term in the per-view budget — the next move on it is a longer
   z lever, or modelling the chamfer properly rather than as one offset.
3. **Capture at least 4–6 views of every board placement.** The multi-view
   estimator needs them, and two views already give a within-placement
   consistency number that catches the gate-escape case (a view passing
   `mad_px < 1.0` while landing 25 mm off in the base frame).
4. **Gate on the chain, not the image.** Views whose board sits >500 px from the
   principal point carry ~8× the tilt error and `mad_px` does not catch them.
   Implemented as `--max-radius` in `fit_board_multiview.py`; fold it into
   capture-time gating so bad views are never taken.
5. **Full DH geometric calibration — later, not now.** The textbook next step
   after joint offsets, but with the post-offset residual correlating with
   nothing (max $|r| = 0.15$) there is nothing configuration-dependent left for
   link-length and $\alpha$ errors to bite on. Revisit once tilt is under
   ~0.3 mm at the corner and robot error is visible above the vision floor.

**Explicitly dropped:** deploying `--flex --droop` in production. Those models
are deleted — see §3.2.
