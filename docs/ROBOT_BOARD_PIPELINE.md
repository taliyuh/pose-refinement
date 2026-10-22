# Stationary cameras to FR3 board pose

Replace `YYYYMMDD` consistently with the session date. The camera-to-camera
rig calibration is reusable and is not repeated while the cameras remain fixed
relative to each other.

## 0. Prepare every new Docker terminal

```bash
source /opt/ros/humble/setup.bash
source /franka_ws/install/setup.bash
```

Only rebuild after code changes:

```bash
cd /franka_ws
colcon build --packages-select calibration_claude --symlink-install
source install/setup.bash
```

## 1. Choose the camera setup

### Single robot-calibrated camera

If only Bolt is available, use the one-camera configuration and keep the
reference camera as `bolt` throughout the pipeline:

```bash
ros2 launch orbbec_camera 4k_bolt.launch.py

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

ros2 run calibration_claude capture_stereo_rgbd \
  --camera bolt:/wrist_camera \
  --output /franka_ws/captures/board_live_YYYYMMDD

cd /home/bartek/pose-refinement
./run_stationary_pipeline.sh \
  --data /home/bartek/franka_ws/captures/board_live_YYYYMMDD \
  --reference-camera bolt \
  --camera-base-calibration \
    bolt=/home/bartek/franka_ws/session_bolt_YYYYMMDD_calibration.json
```

This path does not use `camera_rig_calib_*.json` and does not need a Mega
calibration. The stationary output is directly verified in `bolt_camera`; the
hand-eye file supplies the transform into `base`. The robot and camera must
remain fixed while `capture_stereo_rgbd` acquires the board image.

### Two-camera rig

Run one command in each of two Docker terminals. The launch files contain the
configured camera settings; no command-line camera arguments are required.

```bash
ros2 launch orbbec_camera 4k_mega.launch.py
```

```bash
ros2 launch orbbec_camera 4k_bolt.launch.py
```

## 2. Select or refresh the camera-rig calibration

Use the existing accepted calibration when neither camera has moved relative
to the other:

```text
/home/bartek/pose-refinement/boardpose/out/camera_rig_calib_20260904.json
```

The single board capture in step 5 checks this transform against the current
images. A disagreement makes the refinement unverified, and the exporter then
refuses to create a robot target. One image pair can check an existing rig but
cannot safely estimate a replacement transform.

### Quick transform-only recalibration

Use this path if the same cameras, board, 4K mode, intrinsics, focus, and
exposure settings are retained but one camera may have moved. In Docker,
capture six flat board placements:

```bash
ros2 run calibration_claude capture_stereo_rgbd \
  --output /franka_ws/captures/rig_recalibration_YYYYMMDD
```

Move the board broadly across the shared field of view and press Enter for
each placement. Use at least 10 cm of table translation and at least 45 degrees
of yaw variation. After the sixth capture, type `q` and press Enter.

On the host, create a new candidate without overwriting the last accepted
calibration:

```bash
cd /home/bartek/pose-refinement
.venv/bin/python boardpose/calibrate_stationary_rig.py \
  --data /home/bartek/franka_ws/captures/rig_recalibration_YYYYMMDD \
  --reference-camera mega \
  --recalibrate-transform-from boardpose/out/camera_rig_calib_20260904.json \
  --out boardpose/out/camera_rig_calib_YYYYMMDD_recal01.json
```

The command validates that the prior file is accepted and bound to the exact
current camera streams, freezes its per-camera rim-bias model, and solves only
the new camera transform. It requires at least five accepted synchronized
pairs, verifies every leave-one-out fold, and records the prior file hash and
old-to-new transform delta. It writes nothing if a gate fails. An accepted
candidate becomes active only when its new filename is explicitly supplied in
step 6.

Automatic outer-edge detection may fail on the transparent board. The photos
are already saved, so no recapture is needed. Annotate the saved images with:

```bash
.venv/bin/python boardpose/calibrate_stationary_rig.py \
  --data /home/bartek/franka_ws/captures/rig_recalibration_YYYYMMDD \
  --reference-camera mega \
  --recalibrate-transform-from boardpose/out/camera_rig_calib_20260904.json \
  --out boardpose/out/camera_rig_calib_YYYYMMDD_recal01.json \
  --manual-outlines --restart-outlines
```

Click the four outer corners in any order. If annotation is interrupted, rerun
without `--restart-outlines` to resume.

Expect about two minutes to acquire six placements and up to five minutes of
host processing. On the archived session, the worst-case fallback path took
4 minutes 38 seconds.

### When a full rig calibration is required

Capture nine varied flat placements and omit
`--recalibrate-transform-from` if a camera or lens was replaced, the board was
changed, or image mode, intrinsics, focus, or exposure settings changed. Full
calibration relearns the per-camera rim-bias model and normally needs manual
outer-corner annotation for this transparent board.

Camera movement has two separate consequences:

- Bolt moves relative to Mega: refresh the rig transform only.
- Mega moves relative to the robot base: redo robot-Mega calibration.
- Mega moves relative to Bolt: do both calibrations.
- Both cameras move together as one rigid assembly: the rig remains valid, but
  robot-Mega calibration changes.

## 3. Start FR3 and MoveIt

In another Docker terminal:

```bash
ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:=172.16.0.2 \
  use_fake_hardware:=false \
  load_gripper:=true \
  ee_id:=franka_hand \
  use_rviz:=false
```

## 4. Capture robot-Mega calibration

Attach the configured ArUco marker, then run:

```bash
ros2 run calibration_claude run_taught_capture \
  --poses /franka_ws/poses.txt \
  --cameras /franka_ws/src/calibration_claude/config/cameras_mega.yaml \
  --config /franka_ws/src/calibration_claude/config/calibration_hardware.yaml \
  --frame-id base \
  --out /franka_ws/session_mega_YYYYMMDD \
  --speed-scale 0.20
```

Alternatively, select the calibration views by hand. Enable manual/gravity-free
robot operation using the normal robot procedure, move to a stationary view, and
press Enter for each sample; type `q` to finish:

```bash
ros2 run calibration_claude run_taught_capture \
  --manual \
  --cameras /franka_ws/src/calibration_claude/config/cameras_mega_bolt.yaml \
  --config /franka_ws/src/calibration_claude/config/calibration_hardware.yaml \
  --frame-id base \
  --out /franka_ws/session_manual_YYYYMMDD
```

Manual samples store the measured joints and measured-joint FK TCP pose. If the
process is interrupted, rerun the same command with `--resume`.

If interrupted, resume by adding `--resume` to the same command. Solve the
hand-eye calibration:

```bash
ros2 run calibration_claude solve_handeye \
  /franka_ws/session_mega_YYYYMMDD \
  --camera mega \
  --camera-frame mega_camera \
  --config /franka_ws/src/calibration_claude/config/calibration_hardware.yaml \
  --output /franka_ws/session_mega_YYYYMMDD_calibration.json
```

The result maps the Mega optical processing frame into `base`. Its
`camera_frame` must be `mega_camera`, matching the stationary refiner's output
frame.

When both cameras are present in the same taught capture, produce one joint
calibration artifact instead of two independent hand-eye estimates:

```bash
ros2 run calibration_claude solve_joint_handeye \
  /franka_ws/session_dual_YYYYMMDD \
  --camera mega --camera bolt \
  --camera-frame mega=mega_camera \
  --camera-frame bolt=bolt_camera \
  --config /franka_ws/src/calibration_claude/config/calibration_hardware.yaml \
  --output /franka_ws/session_dual_YYYYMMDD_joint_calibration.json
```

The artifact is a joint fit over both streams with a shared TCP-to-tag mount.
Its final residual is computed by projecting the robot-predicted tag directly
into each captured image and comparing all four corners; PnP is used only for
initialization. The command also performs leave-one-pose-out validation, with
the held-out pose removed from both camera streams before refitting. If any
attempted calibration image lacks a tag detection, validation is marked
incomplete and the stationary pipeline refuses the artifact. The stationary
pipeline can consume a completed artifact for both camera entries by passing
the same file twice:

```bash
./run_stationary_pipeline.sh \
  --data /home/bartek/franka_ws/captures/board_live_YYYYMMDD \
  --reference-camera mega \
  --camera-base-calibration mega=/home/bartek/franka_ws/session_dual_YYYYMMDD_joint_calibration.json \
  --camera-base-calibration bolt=/home/bartek/franka_ws/session_dual_YYYYMMDD_joint_calibration.json
```

## 5. Capture the current board placement

```bash
ros2 run calibration_claude capture_stereo_rgbd \
  --output /franka_ws/captures/board_live_YYYYMMDD
```

Press Enter once to capture, then type `q` and press Enter.

## 6. Refine and export the board pose

Run on the host. Set `RIG_CALIBRATION` to the accepted file chosen in step 2:

```bash
cd /home/bartek/pose-refinement
RIG_CALIBRATION=boardpose/out/camera_rig_calib_20260904.json

./run_stationary_pipeline.sh \
  --data /home/bartek/franka_ws/captures/board_live_YYYYMMDD \
  --reference-camera mega \
  --calibration "$RIG_CALIBRATION"

.venv/bin/python boardpose/export_robot_board_pose.py \
  --calibration /home/bartek/franka_ws/session_mega_YYYYMMDD_calibration.json \
  --refinement boardpose/out/stationary_multicam_board_live_YYYYMMDD.json \
  --capture capture_0001 \
  --output /home/bartek/franka_ws/board_pose_robot_YYYYMMDD.json
```

For a joint artifact, select the reference camera explicitly with
`--camera mega` in the exporter.

The first command jointly fits one board pose against both raw 4K images. The
exporter checks that this joint result is verified, checks frame agreement,
then writes `T_base_board`. The board origin is its centre; the top surface is
`z = 0.025 m` in board coordinates.

If refinement reports `unverified_calibration`, stop: do not export or probe.
Run the quick recalibration in step 2, put the board back at the desired target
placement, repeat step 5, and then rerun this step with the new accepted rig
filename.

## 7. Validate the target without moving

Back inside Docker:

```bash
ros2 run calibration_claude probe_board \
  --board-pose /franka_ws/board_pose_robot_YYYYMMDD.json \
  --offset 0 0 0 \
  --config /franka_ws/src/calibration_claude/config/calibration_hardware.yaml \
  --manipulation-config /franka_ws/src/calibration_claude/config/board_manipulation.yaml \
  --dry-run
```

Offsets are metres in board coordinates. For board +X by 5 cm, use
`--offset 0.05 0 0`.

## 8. Run the live probe

```bash
ros2 run calibration_claude probe_board \
  --board-pose /franka_ws/board_pose_robot_YYYYMMDD.json \
  --offset 0 0 0 \
  --config /franka_ws/src/calibration_claude/config/calibration_hardware.yaml \
  --manipulation-config /franka_ws/src/calibration_claude/config/board_manipulation.yaml
```

Press Enter to approve the approach, Enter during descent to stop, and Enter
again to retract. Keep the physical emergency stop available. The program does
not infer physical contact.
