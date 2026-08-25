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
