# Board pose estimation — 0.2 mm pipeline

Estimates the 6-DoF pose of the NIST ATB board from 4K images, by fitting the
hole-rim contours of `nist_atb_m1.stl` directly to the image with subpixel edge
search, then bundle-adjusting all views with shared camera intrinsics.

Current precision (half-vs-half, no ground truth involved):

| DoF | precision | corner displacement |
|---|---|---|
| in-plane x, y | 0.055 mm | 0.055 mm |
| yaw | 0.012° | 0.044 mm |
| tilt | 0.061° | 0.216 mm |
| along-normal | 0.095 mm | 0.095 mm |
| **combined** | | **0.25 mm** |

End to end in the **robot base frame**, from several shots of one board
placement (`fit_board_multiview.py`, split-half over views, all 11 folders):
**0.208 mm** at the board corner.

Starting point was 3.29 mm / 2.571°, with a p90 tail of 52 mm / 176°.

## Why the old path failed

The previous pipeline reduced ~50,000 dense GLU-Net matches to 4 extrapolated
coplanar corners and ran `solvePnP` on them. Out-of-plane tilt is nearly
unobservable from 4 coplanar points, so **23 of 67 views had 5–51° of tilt
error** — invisible in the match collages, because a homography can be
pixel-perfect over the board interior while placing the corners off the board
entirely. Mask quality and pose quality were decoupled by construction, which
is why GLU-Net inlier counts never predicted accuracy.

## Scope of this document

This is the *why*: the derivation of each stage, the error budget, and the
record of what was tried and rejected. For how to run the thing, see the
[README](../README.md).

Some sections below describe stages that have since been **removed** from the
repository — the GLU-Net seeding, the ArUco path, the on-table 3-DoF fitter, and
several rejected error models. They are kept because the *reasoning* is the
point: each one records a plausible idea and the measurement that killed it, and
that is considerably more useful than the code was. Where a script no longer
exists it is marked **[removed]**.

## Prerequisites

- `.venv` per `requirements.txt`
- Data at `images/images_depth/pose_N/<timestamp>/{color.png,depth.png,pose.txt,joints.json}`

Run everything from the repository root:

```bash
cd /path/to/board_pose
```

## Pipeline

### 0. Capture  (on the robot, needs ROS 2)

Bring the camera up first. It is the wrist Orbbec Femto (`CL8M84100BH`, usb `9-1`),
and `femto_4k.launch.py` is its measurement profile — 4K, manual exposure, manual
white balance, HW depth-to-colour alignment:

```bash
ros2 launch orbbec_camera femto_4k.launch.py
```

It drives the same device as `wrist.launch.py`, so only one may run at a time.
Before the first session, confirm the format survived (see *Camera settings* below):

```bash
ros2 run orbbec_camera list_camera_profile_mode_node   # is NV12 offered at 4K?
ros2 topic hz /wrist_camera/color/image_raw            # must hold ~15 Hz, no drops
```

Then tune the lamps against a real frame before committing to a full session:

```bash
.venv/bin/python boardpose/tune_lighting.py --watch images/images_depth
```

It scores clipping, rim contrast, azimuthal uniformity and pocket-floor level on
each new capture, so a lamp can be moved and the number read back. Aim for zero
clipping at ≥ 254: a clipped rim has no gradient at all, so it is deleted rather
than merely bright, and the previous 47 captures had 0.28–1.70% of pixels there.

```bash
python3 boardpose/capture_board_view.py --root images/images_depth --pose 4
```

Interactive: jog the arm, press ENTER, repeat. Leave it running between views —
a persistent node keeps ROS discovery and the auto-exposure loop warm.

Saves `color.png` / `depth.png` / `pose.txt` exactly as the existing dataset,
**plus `joints.json`** (joint angles, velocities, efforts) and `meta.json`
(settling, sharpness, frame ages). The joint angles are the point: every robot
kinematic model — joint offsets, joint compliance — is a function of them, and
the existing 67 views record only the FK output, so none of it can be fitted.

It also waits out the stick's ringing, refuses frames that predate the keypress,
never resizes, and reports focus sharpness so a defocused pocket floor is caught
at capture time. `s` prints coverage: whether what you have captured can
identify a chain model at all.

**Every capture must be in the same base frame.** `--base-frame` (default
`base_link`) pins it. The original code tried `PLANNING_FRAMES` in order and kept
whichever TF answered first, which varied per session: one run landed in
`base_footprint`, another in `base_link`, another in `right_arm_base_link` —
1.4 m and 135° away. Hand-eye survives it (a constant left-multiplication within
a folder is absorbed by the free per-folder `T_base_board`, the same argument
that makes base tilt unidentifiable), but everything reading `t_base_tcp`
directly does not: the droop model's reach and azimuth, and the coverage gates.
The mixed-frame captures reported a z spread of 1719 mm; the true figure was
226 mm.

Existing captures are fixed exactly — these are static URDF transforms, not a
fit — with:

```bash
.venv/bin/python boardpose/normalize_frames.py          # dry run
.venv/bin/python boardpose/normalize_frames.py --apply  # rewrite
```

Originals are kept as `pose_raw.txt`, the applied transform is recorded in
`meta.json`, and a second run is a no-op.

**Roll the wrist at every camera pose.** The hand-eye offset puts the TCP 540 mm
off the optical axis, so a roll about the optical axis leaves the image nearly
unchanged while swinging the TCP around a circle 1080 mm across. Replaying the
existing 67 poses with simulated rolls, horizontal-reach spread goes 374 → 1098 mm
and azimuth spread 105° → 143° at ±90° alone; 0/90/180/270 adds a little, finer
steps add nothing. Both coverage gates are met without moving the board — which
matters, because a *stationary* board is the normal AX = XB setup and the one
thing a single board position cannot separate (a constant FK bias) is also the one
thing insertion never uses, since `T_tool_hole = T_tool_cam @ T_cam_board @
T_board_hole` never touches the base frame.

#### Camera settings

`femto_4k.launch.py` carries the reasoning in its header. The short version:

- **Format is NV12, not MJPG.** MJPG was measured, not assumed: 8×8 block
  boundaries carry +120% excess horizontal gradient and +33% vertical. But the
  detected rim positions came out uniform mod 8 to within 2–3% over 49,675 points,
  so the σ=1.6 pre-blur in `bundle_rims.load_views` was absorbing it and MJPG was
  never the limiting error. NV12 removes it for free anyway — the fit only ever
  consumes luma, NV12 stores luma full-resolution and uncompressed, and its 2:1
  chroma subsampling is irrelevant because `0.299R + 0.587G + 0.114B` reconstructs
  Y to within 0.04%. At 187 MB/s it also fits USB3 where RGB888's 373 MB/s does not.
  If the device will not offer NV12 at 4K, YUYV then MJPG are both acceptable.
- **Gain floored, exposure long.** Gain amplifies read noise and noise is edge
  error; the board is static and 15 fps allows ~66 ms.
- **Point cloud, colour point cloud and IR off.** Unused, and their bandwidth is
  exactly what an uncompressed 4K colour stream needs.
- **Hole-filling and noise-removal depth filters off.** They invent values, and
  depth is a gate now. Spatial and temporal stay on: static captures mean they
  average noise rather than smear it.
- **Focus is not settable from ROS** (`OB_PROP_COLOR_FOCUS_INT` is in the SDK but
  not wired into the wrapper). It appears not to matter — the bundle fitted
  `fx = 2251.4` against the factory `2252.0`, a −0.02% shift with `cx`/`cy`
  unmoved, across 47 views spanning 402–628 mm of standoff, which a hunting
  autofocus could not survive. Keep the standoff band narrow regardless.
- **`align_mode` is SW, and must be.** This device rejects hardware D2C with
  `Failed to start pipeline: Current stream profile is not support hardware d2c
  process`, then resets and the launch dies — that reset is the "Reset device uid:
  done" you see just before it exits. On firmware 1.0.9 / SDK 2.7.6 it fails for
  *every* combination tried (3840×2160 and 2560×1440 colour, 640×576 and
  1024×1024 depth, MJPG and NV12), so it is the mode itself, not the pairing.
  SW alignment starts with zero errors and does not touch the colour image — it
  only reprojects depth into the colour frame — so intrinsics are unaffected.

Verified output with these settings, which matches the existing dataset exactly:

```
color        3840 x 2160  rgb8    (NV12 on the wire)
depth        3840 x 2160  16UC1   (1024x1024 sensor, SW-reprojected into colour)
camera_info  fx 2251.95  fy 2251.76  cx 1931.43  cy 1091.68
             D [0.0752, -0.1066, -0.0002, 0.0003, 0.0449]
```

Those intrinsics are exactly `K_4K` / `D_4K` as hardcoded in
`boardpose/camera.py`, which confirms the profile is the one the existing
47 views were shot on.

**Exposure is calibrated, not guessed.** At gain 0 against the four lamps:

| `color_exposure` | median grey | clipped ≥ 254 |
|---|---|---|
| 150 | 188 | 0.26 – 4.57 % |
| **60** | **131** | **0.023 %** |
| 30 | 98 | — |

60 is the setting. It puts the diffuse scene on mid-grey with headroom for the
specular glints off the acrylic — and those glints are the reason exposure alone
cannot reach zero: they are 8×8-pixel mirror images of the lamps, and they
saturate at almost any exposure. Driving them out needs diffusion, not a shorter
integration. Lowering exposure cost no contrast: rim contrast measured 15.8 /px
at exposure 60 against 14.7 for the old 47 views.

This matters because clipping lands *on the features*. In the exposure-150
captures, 7–13% of the pocket area was clipped — that is the rims themselves, and
a clipped rim has no gradient for the edge search to find.

> **Gotcha, now fixed:** `ros2 launch` reads the *installed* copy under
> `install/orbbec_camera/share/orbbec_camera/launch/`, which was a plain copy.
> Editing `src/` alone changed nothing, silently — a whole 43-view session was
> shot at exposure 150 while the source said 110. That path is now a **symlink**
> to the source, so edits take effect immediately; `colcon build` will replace it
> with a copy again, so re-check after any rebuild with:
>
> ```bash
> ros2 service call /wrist_camera/get_color_exposure orbbec_camera_msgs/srv/GetInt32
> ```
>
> which reports what the device is *actually* using. A full
> `colcon build --packages-select orbbec_camera` gets OOM-killed on this machine
> at default parallelism — use `--parallel-workers 1`. Launch files are installed
> as data and never compiled, so they never need a rebuild.

Exposure can also be changed at runtime without relaunching:

```bash
ros2 service call /wrist_camera/set_color_exposure orbbec_camera_msgs/srv/SetInt32 "{data: 60}"
```

Lighting, from the residual breakdown: the weakest quartile of edges sits at
0.81 px MAD against 0.36 px for the strongest, so contrast is the dominant driver
and it is a capture problem, not a software one. Four diffused lamps in a
symmetric ring at 60–65° elevation — steep enough to reach the floors of the 12 mm
pockets 22 mm down (which needs > 61°), shallow enough that the specular lobe off
the acrylic top face misses a nadir camera. The four-fold symmetry is the point:
a rim lit brightly on one side has its gradient peak pulled that way, which is a
directional bias no rigid pose can absorb, and is the likely source of the
−0.3…−0.8 px mean residual present in every view.

### 1. Table plane and tilt audit  **[removed: `tilt_vs_depth.py`]**

Fitted the table plane from depth in a ring **around** the board (the board is
transparent, so its own depth is useless) and compared it to each board pose.
Only the plane normal was used, never the offset — a depth scale error maps a
plane to a parallel plane, so it cannot corrupt direction.

This is the diagnostic that exposed the 5–51° tilt errors, and so the reason the
rest of this pipeline exists. It is gone because tilt no longer comes from
depth: the two-level rim model triangulates it from the image (step 4 onward),
and the depth normal is 0.66° off, which is no longer accurate enough to check
against. The intrinsics it carried now live in `boardpose/camera.py`.

### 2. Optional diagnostics  **[removed: `check_flip.py`, `solve_on_table.py`]**

`check_flip.py` confirmed the large tilts were *not* the planar-PnP two-solution
ambiguity (only 4/67 took the wrong branch). `solve_on_table.py` re-solved from
the cached corners with tilt locked to the table; its residual was the first
working quality gate (board-height scatter 25 mm → 1.2 mm).

Both operated on the old 4-corner cached poses, which no longer exist. The
surviving descendant of the flip check is `flip_recover.py`, which detects and
*repairs* 180° yaw flips in the current rim fits.

### 3. Inspect the model features

```bash
.venv/bin/python boardpose/board_model.py
```

Extracts from the STL (which is in **centimetres**: 320 × 250 × 25 mm):

- 20 hole rims on the top face at z = 25 mm
- 20 pocket floor rims at z = 3 mm — the holes are **blind pockets 22 mm deep**,
  and that 22 mm baseline is what makes tilt observable
- ~1184 points per level at 1 mm arc spacing

The outer rectangle is deliberately excluded: 25 mm of transparent acrylic makes
it a double edge (top and bottom faces up to 16 px apart at 4K).

### 4. Per-view rim fit  (~3 min)

```bash
.venv/bin/python boardpose/fit_rims6.py --out boardpose/out/rims6.json
```

Full 6-DoF, seeded per image by `seed_board.py` from the image and its depth
frame alone — no marker, no cached pose. Each of the top-20 seed candidates is
refined and the winner chosen on the **rim residual**, the quantity actually
being minimised, rather than on the chamfer score that ranked them.

Quality gate: `mad_px < 1.0`. This replaces every heuristic mask-quality metric —
those all sat at chance (AUC 0.52–0.55) because they scored the mask, not the
pose.

> **Superseded: `fit_dense_on_table.py` [removed].** It solved only yaw, u, v,
> taking roll, pitch and height from a depth plane whose ROI came from the seed
> pose — so a wrong seed produced a plane fitted in the wrong place and no
> in-plane search could recover. Measured on `pose_12/131731_390724`: widening
> the search from ±30 mm to ±150 mm with full 360° yaw moved the residual only
> 1.72 → 1.52 px, while freeing all 6 DoF reached 0.46 px. The table constraint
> was a workaround for 4-corner PnP being blind to tilt; 1184 subpixel points
> over two contour levels are not, so the constraint bought nothing and cost the
> dependency on depth.

### 5. Bundle adjustment  (~70 s)

```bash
.venv/bin/python boardpose/bundle_rims.py \
    --out boardpose/out/bundle_2level.json
```

**This is the estimator.** Joint solve over all views:

- shared: `fx fy cx cy k1 k2 p1 p2 k3`
- per view: full 6-DoF pose, **tilt free** (depth is no longer an input)
- ~50k subpixel rim observations, robust loss, sparse Jacobian

Points are projected *into the raw image*, so distortion is estimated on
unresampled pixels. This is Zhang calibration performed on the actual
measurement object rather than a separate checkerboard.

Useful flags:

```bash
--top-only         # single-plane model, for comparison
--fix-intrinsics   # hold the camera fixed
--max-mad 1.5      # seeding threshold from step 4
```

Writes `out/bundle_2level.json` (refined intrinsics + per-view `T_cam_board`).

### 6. Visual check

```bash
.venv/bin/python boardpose/render_fit.py
```

Writes 47 panels to `out_overlays/`. Left: full frame with green top rims, cyan
pocket floor rims (only where not occluded by the pocket wall), and the **old
4-corner pose outline in red**. Right: native-resolution zooms on four pockets.

If the green contours sit on the rims and the red outline does not, the fit is
good. `fit_pose1_104121_241212.png` is a clear example — you labelled that mask
pixel-perfect and its 4-corner pose was 18.8° out.

### 7. Hand-eye calibration from the board  (~20 s)

```bash
.venv/bin/python boardpose/handeye_board.py
```

Solves `T_base_tcp(i) @ X @ T_cam_board(i) = T_base_board(p)` for the hand-eye
`X = T_tcp_cam` (shared) and the board pose per folder. The camera is on a stick
held by the gripper and the board is static within a folder — that is exactly
the AX = XB setup, with the board replacing the ArUco marker as the target.

Runs both the marker-based and board-based calibration and scores them on the
*same* task, so the marker is not penalised for sitting 250 mm from the board.
Residuals are reported as board-corner displacements in mm, and validated
against the touch-probed corners.

Result: `X` translation `[+32.5, -470.2, +320.2] mm`, `|t|` = 570 mm — which
matches the physical stick length. Also writes `out/handeye.json`.

**No deflection model is offered here.** Three have now been tried and all
three are rejected — see step 8b, which rejected the third with a placebo
control after leave-one-folder-out had passed it 11/11. All three are deleted;
`handeye_board.py` keeps the post-mortem as a comment so they are not
reinvented. `--flex` bought 11% for 18 parameters and moved `X` by 22 mm.
**droop** rotated the FK output about a horizontal axis through the shoulder by
an angle proportional to reach — the actual physics in one parameter — and gave
2.19 → 2.17 mm with an unchanged residual-vs-reach correlation and a physically
absurd fitted rate of 1.18 °/m.

Why neither could work: **these captures reach upward, not outward.**
`|t_base_tcp|` is 1252–1522 mm but the *horizontal* reach is only 232–711 mm, so
the gravity moment about the shoulder barely varies. The "+0.68 correlation with
reach" once quoted here is a correlation with `|t|`, dominated by TCP height
rather than lever arm. The right model is joint offsets, which needs the joint
angles the old dataset never recorded — hence step 0, and step 9.

A base-not-level rotation is **not** offered: it is a constant left-multiplication
of the FK, so it is exactly cancellable by rotating the free per-folder
`T_base_board`, and fitting it anyway sends the base to 55° while still lowering
the residual. Unidentifiable against this objective, full stop.

### 6b. Rim bias — where the tilt error was

`bundle_rims.py --rim-bias b0` (the default) fits a constant radial offset per
contour level, on top of the intrinsics. **The STL outline is not where the
intensity edge is.** Measured over 144 views with the bundled poses, the signed
rim residual is −0.733 px on the top rims (z = 25 mm, chamfered) and +0.306 px
on the pocket floor rims (z = 3 mm, at the base of a 22 mm wall) — the same sign
in **144 of 144 views**.

Tilt is triangulated from the 22 mm baseline between those two levels, so the
*differential* between them is a tilt error by construction:

- its **mean**, 1.25 px, is 0.68° of constant tilt bias — which is what made the
  1-plane and 2-level models disagree by 0.224°;
- its **scatter**, sd 0.333 px, is 0.181° of tilt — and the measured per-view
  tilt error was 0.182°.

Two parameters, fitted inside the bundle. What they are worth:

| | no rim bias | `--rim-bias b0` |
|---|---|---|
| rim residual MAD | 0.684 px | **0.265 px** |
| tilt, per view | 0.183° (0.722 mm) | **0.141° (0.557 mm)** |
| chain residual, joint offsets | 0.865 mm | **0.762 mm** |
| leave-one-folder-out | 0.787 mm | **0.645 mm** |
| multi-view split-half | 0.314 mm | **0.208 mm** |
| folders beating pose averaging | 7 / 11 | **11 / 11** |

The offsets are in **mm on the board**, not pixels: the residual tracks standoff
with the sign and size of a fixed board-plane offset (top +0.0015 px/mm, floor
−0.0013 px/mm over 340–554 mm), which is a geometric effect, not blur.

**`--rim-bias b0sym` — safe, and useless.** The symmetric view terms
(`b2·|s|`, `b3·|u_xy|`) were added to catch the ~0.24 mm of orientation-dependent
error that step 8b proved is not in the arm. They are built so they change rim
*size*, not position — verified: for a 1 mm coefficient the centroid moves
0.007 mm against 0.17 mm of radial change, where `b1` moves the centroid
0.109 mm and the radius not at all. They fit sensible values and improve the
picture. They do not improve the answer:

| | `b0` | `b0` + `b2,b3` |
|---|---|---|
| rim residual MAD | 0.265 px | **0.250 px** better |
| chain residual | 0.762 mm | 0.757 mm flat |
| leave-one-folder-out | **0.645 mm** | 0.648 mm worse |
| multi-view split-half | **0.208 mm** | 0.221 mm worse, 7/11 folders |

Four parameters, a better image fit, a worse pose. **The pixel residual is not
the objective.** What it rules out matters more than what it cost: the
orientation-dependent error is *not* a rim radius or shape effect. It behaves
like a genuine per-view pose perturbation correlated with camera orientation,
which points at the intrinsics or the arm — look there, not at the board model.

**`--rim-bias b0b1` exists and should not be used.** The view-geometry term
`b1·s` offsets each point by `b1·(u_xy·n)`, and since `n` sweeps every direction
around a closed rim while `u_xy` barely changes, that is very nearly a per-view
in-plane *translation* of the board — a pose parameter. Fitted, it runs to
1.74 and 2.11 mm, an order of magnitude past the effect it models, for 0.006 px.
Same failure mode as flex/droop. `--rim-bias none` restores the old model.

### 9. Board pose from N shots  (~2 min for 15 views)

```bash
.venv/bin/python boardpose/fit_board_multiview.py --pose 12 --split-half
```

The production estimator when you have several captures of one static board.
Fits **one** `T_base_board` against every view's subpixel rim edges at once,
with a per-view 6-DoF chain perturbation under a Gaussian prior of width
`--chain-sigma` (default 0.40 mm, chosen by `eval_multiview.py --sweep`).

That prior is the whole design. The per-view error budget is

| source | per view |
|---|---|
| vision, in the camera frame | 0.13 mm (rim mad 0.63 px) |
| the robot chain, after joint offsets | 0.87 mm |

so the chain is **7× noisier than the vision**. Tying every view rigidly to one
board pose asserts the opposite, forces 0.87 mm of chain error into the board
pose, and is measurably worse than the pose averaging it replaces (pose_12
split-half 0.788 mm vs 0.524 mm). With the slack term the same fit reaches
0.392 mm — and rim residuals return to 0.6 px, i.e. the misfit really was the
chain, not the board.

Split-half over the views, all 11 folders:

| | joint fit | averaging |
|---|---|---|
| median | **0.208 mm** | 0.360 mm |
| worst folder | **0.465** | 1.444 |
| folders improved | **11 / 11** | |

`--chain-sigma` was picked on this same statistic, so treat 1.73× at the median
as an upper bound rather than an independently validated figure.

Also gates on `--max-radius` (default 500 px): views whose board sits far off
the principal point carry ~8× the tilt error, and `mad_px` does not catch them.

```bash
.venv/bin/python boardpose/eval_multiview.py           # every folder
.venv/bin/python boardpose/eval_multiview.py --sweep   # scan chain-sigma
```

Split-half over the *views*: fit two disjoint halves independently, compare.
The board never moved, so the disagreement is the error, doubled — and both
estimators see the same views, so it is a comparison of models, not of data.

### 8b. Configuration-dependent hand-eye — tested, rejected  **[removed: `test_wrist_flex.py`]**

After the rim-bias fix, `theta_6` became the strongest predictor of the leftover
residual (+0.337, against 0.15 for everything else), suggesting a hand-eye that
varies with wrist roll — the camera hangs on a stick, and rolling it swings the
mass. Tested with leave-one-folder-out as the gate:

| model | params | LOFO median | folders improved |
|---|---|---|---|
| joint offsets only | 4 | 0.645 mm | — |
| + wrist flex (θ₆) | 16 | 0.515 | 11/11 |
| + gravity flex (TCP) | 22 | **0.406** | 11/11 |
| + **PLACEBO** fake_x | 22 | **0.408** | 11/11 |
| + **PLACEBO** fake_y | 22 | 0.452 | 11/11 |

**A direction with no gravitational meaning scores identically to real gravity.**
In this dataset the camera always points at the board, so the arm's orientation
lives on a narrow manifold and every "fixed direction in the TCP frame" is
nearly the same regressor — canonical correlations between the gravity and
fake_x feature sets are 0.997 and 0.986. The data cannot separate gravity-driven
sag from any smooth function of camera orientation.

**The lesson: leave-one-folder-out is not a sufficient gate on its own.** It
shows a parameter transfers between folders; it cannot show the parameter means
what its name says. Every folder here uses similar arm configurations, so a
view-dependent *vision* error transfers exactly as well as a robot property
would. Run the placebo.

What is real: ~0.24 mm of the residual is a reproducible function of camera
orientation, common to every folder, and it is **not in the arm** — the fitted
amplitude is 2 mm at the board on a 135 mm stick, and it grows without plateau
as the amplitude prior is loosened (0.1 mm → 0.624, 0.3 → 0.543, 1.0 → 0.406,
3.0 → 0.352). It belongs in the vision model.

### 8. Validation (no ground truth used)

```bash
.venv/bin/python boardpose/tilt_precision.py   # tilt precision, 1-plane vs 2-level
.venv/bin/python boardpose/eval_multiview.py   # end-to-end split-half, all folders
```

`tilt_precision.py` fits each view twice from disjoint halves of the pockets and
compares — pure internal precision, nothing external can flatter it.
`eval_multiview.py` does the same thing one level up, over disjoint halves of
the *views*, which is the honest end-to-end number quoted at the top.

**[removed]** `split_half.py` measured in-plane precision the same way on the old
on-table 3-DoF fits; `eval_consistency.py` checked that `T_marker_board` was
constant within a folder. Both were tied to stages that no longer exist — the
first to the on-table fitter, the second to the ArUco marker.

The probed ground truth in `ground_truth.txt` is accurate to only ~1.5–2.5 mm
(side lengths come out 315–323 mm against 320 nominal), so it can confirm a fix
but cannot certify 0.2 mm. Do not tune against it.

## Known issues

- **56 mm offset.** The probed board top face sits at z ≈ 808 mm in the robot
  base frame, not 727 + 25 = 752. The probe offset itself is correct (fitted
  341.8 mm vs coded 342.4). Unresolved; harmless for relative work.
- **Tilt accuracy.** *Largely resolved* — see step 6b. The 1-plane vs 2-level
  disagreement of 0.224° was the differential rim bias between the two contour
  levels; fitting it removed 0.68° of tilt bias and cut per-view tilt from
  0.183° to 0.141°. What remains (0.141°, 0.557 mm at the corner) is still the
  largest single term in the per-view budget.
- **ArUco marker.** Contributes ~1.0 mm / 0.38°, unchanged across every board
  estimator tried. Anything routed through it inherits that.
- **Depth.** Its table normal is 0.66° off the image-derived one. Keep it for
  initialisation and gating; it is no longer accurate enough to set tilt.
- **The robot chain is now the dominant error.** Board pose is good to 0.25 mm
  in the camera frame, but after `T_base_tcp @ X` the same static board scatters
  by 2–5 mm across views. That scatter is *configuration-dependent, not random*:
  in pose_1 it correlates +0.68 with arm reach, and in pose_3 the views at
  ~1610 mm reach are displaced a consistent +4 mm in x versus those at
  ~1485 mm. Hand-eye is not the cause — the marker-derived and board-derived
  `X` agree to 7 mm in 570 (1.2%) and 0.6°.
- **Gate escape.** `pose_3/112248_360185` passes `mad_px < 1.0` (0.90) yet lands
  25 mm off in the base frame. The rim residual gates the *vision*, not the
  chain; add a cross-view consistency check when combining views.

## Removed

Kept as a list so none of it gets reinvented.

**Failed models.** Each surviving file keeps a comment saying what was tried and
why it failed, which is the only part worth having.

- `score_views.py`, `eval_metrics.py`, `eval_photometric.py` — heuristic
  mask-quality metrics, all at chance (AUC 0.52–0.55). They scored the mask, not
  the pose; the rim residual replaced all of them.
- `handeye_board.py --flex` / **droop**, and the compliance term in
  `kinematic_calib.py` — see steps 6 and 7.
- Configuration-dependent hand-eye — see step 8b, and run the placebo.
- `--rim-bias b0b1` and `b0sym` — see step 6b. Better pixel residual, worse pose.

**Superseded stages**, removed when the repository was prepared for release.
The reasoning for each is in the step above that describes it.

| removed | replaced by |
|---|---|
| GLU-Net dense matching + 4-corner PnP (`estimate_glunet.py`, `run_refinement_all.py`, `glunet_src/`) | `fit_rims6.py` — direct rim fitting |
| GTSAM factor graph (`solve_poses_gtsam.py`, `run_depth_refinement.py`) | `bundle_rims.py` — sparse bundle adjustment |
| ArUco seeding (`update_cached_views_all.py`, `calib.py`, `aruco_localise.py`) | `seed_board.py` — marker-free, from image + depth |
| on-table 3-DoF fit (`fit_dense_on_table.py`) | `fit_rims6.py` — full 6-DoF, tilt free |
| depth tilt audit (`tilt_vs_depth.py`) | the two-level rim model; intrinsics moved to `camera.py` |
| `check_flip.py`, `solve_on_table.py`, `split_half.py`, `eval_consistency.py` | `flip_recover.py`, `eval_multiview.py` |
| `cached_views.json` and its maintenance scripts | `joints.json`, written at capture time |
| BlenderProc synthetic data (`generate_data.py`) | never used by the metrology path |

The one thing worth restating: **the GLU-Net stage was not slow or fragile, it
was solving the wrong problem.** Reducing 50,000 dense matches to 4 coplanar
corners throws away exactly the information that constrains tilt, and no amount
of match quality can put it back. Mask quality and pose quality were decoupled
by construction.
