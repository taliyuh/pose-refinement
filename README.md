<!-- markdownlint-disable MD013 -->
# board_pose

Measures the 6-DoF pose of a NIST ATB task board in a robot's base frame, to
**0.208 mm** at the board corner, from a handful of photos taken by a camera on
the arm.

No fiducial markers, no CAD alignment by hand, no depth camera in the loop. The
board's own hole rims are fitted directly to the image with subpixel edge
search, and every view is bundle-adjusted together.

| what | precision |
|---|---|
| in-plane x, y | 0.055 mm |
| yaw | 0.012° |
| tilt | 0.061° |
| along-normal | 0.095 mm |
| **end to end, in the robot base frame** | **0.208 mm** |

Measured by split-half (fit two disjoint halves of the views, compare). No
ground truth is involved, so nothing here can be flattered by tuning.

---

## Install

```bash
git clone <this repo> && cd board_pose
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

That is all the refinement pipeline needs — pure numpy/scipy/OpenCV, no GPU.
Capture additionally needs ROS 2 (`rclpy`, `sensor_msgs`, `tf2_ros`), which
comes from your system install rather than pip.

## The workflow

### Stationary single-camera workflow

The stationary refiner also supports one robot-calibrated camera. Use the
camera YAML for the camera that is present; for Bolt, the checked-in config
uses the `/wrist_camera` topics:

```bash
ros2 run calibration_claude run_taught_capture \
    --poses /franka_ws/poses.txt \
    --cameras /franka_ws/src/calibration_claude/config/cameras_bolt.yaml \
    --config /franka_ws/src/calibration_claude/config/calibration_hardware.yaml \
    --out /franka_ws/session_bolt_YYYYMMDD

ros2 run calibration_claude solve_handeye \
    /franka_ws/session_bolt_YYYYMMDD \
    --camera bolt \
    --camera-frame bolt_camera \
    --config /franka_ws/src/calibration_claude/config/calibration_hardware.yaml \
    --output /franka_ws/session_bolt_YYYYMMDD_calibration.json
```

Capture only that camera for the stationary board measurement:

```bash
ros2 run calibration_claude capture_stereo_rgbd \
    --camera bolt:/wrist_camera \
    --output /franka_ws/captures/board_live_YYYYMMDD

./run_stationary_pipeline.sh \
    --data /franka_ws/captures/board_live_YYYYMMDD \
    --reference-camera bolt \
    --camera-base-calibration \
        bolt=/franka_ws/session_bolt_YYYYMMDD_calibration.json
```

No camera-rig calibration or second `--camera-base-calibration` argument is
needed. The result is written in `bolt_camera` and can be composed with the
robot base using `export_robot_board_pose.py` as usual. Keep the camera and
the robot stationary during this board capture; the hand-eye artifact is a
fixed camera-to-base calibration.

### Stationary two-camera rig

Start the two cameras in separate container terminals.  The launch files carry
the measured exposure defaults (Mega 110, Bolt 26), so no command-line camera
overrides are needed:

```bash
ros2 launch orbbec_camera 4k_mega.launch.py
ros2 launch orbbec_camera 4k_bolt.launch.py
```

Capture a dedicated rig-calibration session from a third container terminal:

```bash
source /opt/ros/humble/setup.bash
source /franka_ws/install/setup.bash
ros2 run calibration_claude capture_stereo_rgbd \
    --output /franka_ws/captures/pose_refinement_rig_YYYYMMDD
```

Leave both cameras fixed.  Move the board flat across the table and press Enter
once at each placement.  Vary x/y position and yaw as widely as both cameras
allow; nine placements are recommended and eight are required.  The capture
node waits for the scene to settle and obtains one fresh RGB-D pair from each
camera independently.  The board must remain stationary until both views have
arrived; their inter-camera timestamp span is recorded for diagnostics but is
not used as an acceptance gate.

For captures laid out as `capture_NNNN/<camera>/` (as in `run_01`), use the
stationary-rig entry point instead of the single-camera workflow:

```bash
./run_stationary_pipeline.sh --data run_01 --reference-camera mega
```

It fits every native **3840 x 2160** image using that camera's own
`camera_info.json`, estimates the fixed transform from `mega` to `bolt` from
the synchronized pairs, and writes board poses in the selected reference
camera frame.  There is intentionally no robot-base transform in this result:
apply your later camera-to-base calibration as a left multiplication.

The command also writes `boardpose/out_stationary_overlays/<dataset>/capture_NNNN.png`,
native-resolution `capture_NNNN_<camera>_4k.png` files, and `scene_3d.png`.
Each image pair has the fitted top rims and board outline in green, and the
visible pocket-floor rims in cyan. In a calibrated run, magenta/orange also
shows the local image refinement: it should closely overlap the shared
green/cyan rig pose. Verify these contours at native resolution before using a
pose. The 3D scene shows the fixed camera rig and every fitted board pose in
the reference-camera frame. A rig estimated from just one synchronized pair is
reported as unverified. One current pair can, however, verify a previously
accepted fixed rig; export is blocked if that check fails.

To make a reusable calibration from a dedicated, varied capture set, run:

```bash
.venv/bin/python boardpose/calibrate_stationary_rig.py --data run_calibration \
    --reference-camera mega --out boardpose/out/camera_rig_calib.json
```

This first checks full-4K outer-board rectangles and writes annotated previews
next to the calibration JSON. It records the fixed camera transform,
camera-info provenance, capture timing, per-camera top/floor rim-bias terms,
and accepted or rejected pairs. Eight varied pairs are required to identify
the rim-bias terms; nine or more give the leave-one-out validation useful
redundancy. The outer rectangle deliberately supplies all four cyclic pose
hypotheses, so the unmarked rectangle's yaw convention cannot contaminate the
rig transform.

Transparent acrylic and clutter can make automatic rectangle detection reject
an otherwise valid capture. In that case, use the same existing photos and
click the four outer corners in each image for at least eight synchronized
pairs:

```bash
.venv/bin/python boardpose/calibrate_stationary_rig.py --data run_calibration \
    --reference-camera mega --out boardpose/out/camera_rig_calib.json \
    --manual-outlines --restart-outlines
```

Corners may be clicked in any order. The calibration preserves progress in
`camera_rig_calib.outline_detections.json`; omit `--restart-outlines` when
resuming a partially completed manual session. Use it when replacing a failed
automatic pass, as shown above. The clicked outline only initializes the
calibration. The default final pass independently fits each reference image at
full resolution, uses that pose only to seed an unconstrained local refinement
of the synchronized secondary view, learns a bounded constant top/floor rim
bias for each camera, then freezes those biases while jointly fitting the rig
and one board pose per capture. This is much less sensitive to the ambiguous
acrylic outer edge and prevents a camera-specific rim bias from being absorbed
into the inter-camera transform.

Because the board is constrained to the table, calibration uses broad yaw and
table-position coverage rather than requiring tilt. The output is accepted
only after optimizer, rank, residual, capture-count, intrinsic-provenance,
timing, and leave-one-out checks pass.

If the same cameras and image settings are retained but one mount moved, use
the six-placement transform-only recalibration procedure in
[`docs/ROBOT_BOARD_PIPELINE.md`](docs/ROBOT_BOARD_PIPELINE.md). It reuses the
validated per-camera rim biases while solving and validating a fresh rig
transform. A camera, lens, board, image-mode, focus, exposure, or intrinsics
change still requires the full calibration above.

Use that saved rig for measurements. In this mode `mega` is fitted normally,
then its board pose is projected into `bolt` through the fixed rig transform.
The local `bolt` refinement is retained as a rig-consistency diagnostic; the
reported result comes from one shared board pose fitted against both raw 4K
images. The overlay is drawn from that shared pose in both cameras, so a bad
independent `bolt` seed cannot replace the result.

```bash
./run_stationary_pipeline.sh --data run_02 --reference-camera mega \
    --calibration boardpose/out/camera_rig_calib.json
```

### 1. Take photos

Move the arm around the board and shoot. Leave the script running between
views — it keeps ROS discovery and the exposure loop warm.

```bash
python3 boardpose/capture_board_view.py --root images/images_depth --pose 4
```

Jog the arm, press ENTER, repeat. 12–15 views is plenty. Each capture writes:

```
images/images_depth/pose_4/<timestamp>/
    color.png    depth.png    pose.txt    joints.json    meta.json
```

`joints.json` is the one that matters and the one most datasets forget. Every
robot kinematic model is a function of the joint angles; logging only the FK
output means none of it can ever be fitted.

Three things to get right, in order of how much they cost you:

- **Roll the wrist at every camera pose.** The camera sits off the tool axis, so
  a roll about the optical axis barely changes the image while swinging the TCP
  around a wide circle. That is what makes the hand-eye observable. Without it
  the calibration is degenerate and you will not be able to tell.
- **Keep every capture in one base frame.** `--base-frame` (default `base_link`)
  pins it. If you have older captures in mixed frames, `normalize_frames.py`
  repairs them from static URDF transforms — `--apply` to rewrite, no argument
  for a dry run. Originals are kept as `pose_raw.txt`.
- **Do not clip the highlights.** A clipped rim has no gradient at all, so it is
  deleted rather than merely bright. Aim for zero pixels at ≥ 254:

  ```bash
  .venv/bin/python boardpose/tune_lighting.py --watch images/images_depth
  ```

  It scores clipping, rim contrast and pocket-floor level on each new capture,
  so you can move a lamp and read the number back. Four diffused lamps in a
  symmetric ring at 60–65° elevation is the arrangement that works: steep enough
  to light the floors of the 22 mm-deep pockets, shallow enough that the
  specular lobe misses a nadir camera.

### 2. Run the pipeline

```bash
./run_pipeline.sh --pose 4
```

Roughly 30–60 min for 15 views, most of it in the per-view fit. It is resumable:
rerunning picks up where a killed run stopped.

Out comes `boardpose/out/board_pose_4.json`, and on stdout:

```
T_base_board =
  translation [ -918.306  +330.953 +1552.194] mm
  rotation    [ +105.351  -102.049   +53.998] deg

  half-vs-half disagreement (2x the precision of each half):
    joint multi-view fit    :  0.328 mm at the corner
```

That last number is your error bar, computed from your own data. If it is much
worse than ~0.4 mm, see *Troubleshooting*.

**First run on a new robot needs `--calibrate`** — see below.

### 3. Look at it (optional)

```bash
.venv/bin/python boardpose/render_fit.py --src boardpose/out/bundle.json \
    --outdir boardpose/out_overlays
```

One panel per view: the full frame with the fitted rims drawn in green (top
rims) and cyan (pocket floors), plus native-resolution zooms on four pockets. If
the contours sit on the rims, the fit is good. This is the check that actually
catches problems — a pose can be tens of millimetres out while every summary
statistic looks fine.

---

## Calibrating for your robot

The pipeline needs two things that belong to *your* hardware, not to the board:

| | what it is | how often |
|---|---|---|
| **camera intrinsics + rim bias** | `bundle.json` | per camera |
| **joint offsets + hand-eye** | `kinematic_calib.json` | per arm and camera mount |

Both are produced by `--calibrate`, which fits them across **several different
board placements** (not several views of one placement — the placements are what
separate a robot error from a board error):

```bash
# capture 8-11 folders, moving the board between each
python3 boardpose/capture_board_view.py --root images/images_depth --pose 5
python3 boardpose/capture_board_view.py --root images/images_depth --pose 6
# ...

./run_pipeline.sh --pose 4 --calibrate
```

After that, every later measurement is just `./run_pipeline.sh --pose N`.

The repo ships the calibration from the reference robot (a UR5) so the pipeline
runs out of the box, but **those numbers describe our arm, not yours.** Run
`--calibrate` before trusting any absolute figure.

### A different arm

The reference data was shot on a **UR5**. The arm model lives in
`boardpose/kinematics.py` and is a plain DH table:

```python
DH = {
    "ur3": (...), "ur5": (...), "ur10": (...),
    "ur3e": (...), "ur5e": (...), "ur10e": (...), "ur16e": (...),
}
ACTIVE_ROBOT = "ur5"
```

- **Another UR:** set `ACTIVE_ROBOT`, or pass `robot=` to `fk()`.
- **Any other 6R arm:** add its `(d, a, alpha)` triple under a new name. Nothing
  else in the pipeline knows or cares which arm it is.
- **Not expressible in standard DH** (7-DoF, unusual chain): use `fit_poe()` at
  the bottom of `kinematics.py`, which identifies a generic
  product-of-exponentials chain instead. It needs a reasonable seed.

You do not have to take the model on faith. Run:

```bash
.venv/bin/python boardpose/kinematics.py
```

`T_base_arm = T_base_tcp @ inv(fk(theta))` is the constant base mounting
transform, so if the model is right it is *identical* for every capture, and its
scatter over the captures is exactly the model error. On our data:

```
arm identification (base-transform scatter, lower is better):
    ur5        0.176 mm   <- this arm
    ur5e       2.972 mm
    ur16e     12.633 mm
    ur10e     58.242 mm
    ...
```

A correct model sits well under a millimetre. If nothing does, your DH table is
wrong — no amount of downstream fitting will rescue that.

Also check the joint names. `boardpose/kinematics.py` sets

```python
JOINT_PREFIX = "right_arm_"     # ours is a dual-arm rig
```

A bare UR driver publishes `shoulder_pan_joint` etc., so set `JOINT_PREFIX = ""`.
A wrong prefix shows up as `captures with joint data: 0`, never as quietly
wrong numbers.

### A different camera

`boardpose/camera.py` holds `K_4K` and `D_4K` — the factory intrinsics of the
reference camera at 3840×2160. Replace them with yours from `camera_info` or any
checkerboard calibration. They only need to be close enough for the first fit to
converge: `bundle_rims.py` re-estimates `fx fy cx cy k1 k2 p1 p2 k3` across all
views against ~50k rim observations, and everything downstream reads the refined
values out of the bundle.

That is Zhang calibration performed on the actual measurement object instead of
a separate checkerboard, which is what removes the systematic bias no amount of
per-view fitting can reach.

---

## Troubleshooting

**`captures with joint data: 0`** — `JOINT_PREFIX` does not match your driver.

**Split-half much worse than 0.4 mm** — run step 3 and look at the overlays.
Usually one or two views are grossly wrong rather than everything being slightly
wrong.

**A view is ~410 mm off** — that is the 180° yaw flip. The board is nearly
2-fold symmetric, so the search has a second, almost-as-good basin. It is
repaired automatically in step 2 of the pipeline; `flip_recover.py --detect-only`
reports without changing anything.

**Refusing `--refine-x`** — the wrist did not rotate enough between views for the
hand-eye to be identifiable. Recapture with wrist roll (see step 1).

**A view passes the residual gate but lands far off in the base frame** — the rim
residual gates the *vision*, not the robot chain. This is a known gap; the
multi-view fit's per-view slack term absorbs most of it, and `--max-radius`
drops views whose board sits far from the principal point (those carry ~8× the
tilt error and the residual does not catch them).

## Known limits

- **Tilt is the largest remaining term** — 0.141°, or 0.557 mm at the corner.
- **The robot chain, not the vision, is now dominant.** Board pose is good to
  0.25 mm in the *camera* frame; the arm contributes ~0.87 mm per view. The
  multi-view estimator is built around exactly this asymmetry.
- **~0.24 mm of orientation-dependent error is unexplained.** It is reproducible,
  common to every folder, and demonstrably *not* in the arm — see
  `docs/METHOD.md`, which also records the placebo control that killed three
  plausible-looking models for it.
- **Depth is only used for seeding and gating.** Its table normal is 0.66° off
  the image-derived one, which is no longer good enough to set tilt.

## Layout

```
run_pipeline.sh          captures in, board pose out
nist_atb_m1.stl          the board model
boardpose/
    capture_board_view.py    step 1: capture, with joint angles
    tune_lighting.py         step 1: lamp scoring
    normalize_frames.py      step 1: repair mixed base frames
    camera.py                intrinsics + subpixel edge search
    board_model.py           STL -> hole-rim contours
    seed_board.py            marker-free seed pose from one image
    fit_rims6.py             per-view 6-DoF rim fit
    flip_recover.py          180-degree yaw flip repair
    bundle_rims.py           THE ESTIMATOR: joint bundle adjustment
    kinematics.py            arm model (DH table + arm identification)
    handeye_board.py         hand-eye from the board itself
    kinematic_calib.py       joint offsets + hand-eye
    fit_board_multiview.py   THE ANSWER: one T_base_board from N views
    eval_multiview.py        split-half validation, all folders
    render_fit.py            step 3: overlays
    tilt_precision.py        tilt precision, 1-plane vs 2-level
    board_scatter.py         per-folder consistency (used by flip_recover)
docs/METHOD.md           why it is built this way, and what did not work
```

## Further reading

`docs/METHOD.md` is the long version: the derivation of each stage, the error
budget, and — more useful — the record of what was tried and rejected, with the
measurements that rejected it. Three separate deflection models, a
configuration-dependent hand-eye, and two extra rim-bias terms all looked good
and all failed. The placebo control that caught the last one is the most
transferable thing in this repository.
