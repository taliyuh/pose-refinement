#!/usr/bin/env bash
# run_pipeline.sh - captures in, board pose out.
#
#   ./run_pipeline.sh --pose 12                 measure pose_12
#   ./run_pipeline.sh --pose 12 --calibrate     re-fit the robot first (slow)
#
# Stops at the first failing step, so a bad fit never reaches the pose.
# Everything is relative to this file: no absolute paths, no PID waiting.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
BP="$ROOT/boardpose"
OUT="$BP/out"

POSE=""
CALIBRATE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pose)      POSE="$2"; shift 2 ;;
        --calibrate) CALIBRATE=1; shift ;;
        -h|--help)   sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)           echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$POSE" ]] || { echo "--pose N is required" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "no interpreter at $PY (set PYTHON=...)" >&2; exit 2; }

mkdir -p "$OUT"
say()  { echo "[$(date +%H:%M:%S)] $*"; }
step() { say "--- $1 ---"; shift; "$@"; }

# 1. Per-view 6-DoF rim fit. Seeds itself from each image + its depth frame.
#    Resumable: rerunning picks up where a killed run stopped.
step "per-view rim fit" \
    "$PY" "$BP/fit_rims6.py" --out "$OUT/rims6.json"

# 2. Repair 180-degree yaw flips before they reach the bundle. The board is
#    nearly 2-fold symmetric, so a few views land in the wrong basin; the wrong
#    basin fits measurably worse, which is what makes them recoverable.
#    Writes --out only when it actually corrected something, hence the fallback.
step "flip repair" \
    "$PY" "$BP/flip_recover.py" --src "$OUT/rims6.json" --out "$OUT/rims6_fixed.json"
FIT="$OUT/rims6_fixed.json"
[[ -f "$FIT" ]] || FIT="$OUT/rims6.json"

# 3. THE ESTIMATOR. Joint solve over every view: shared intrinsics + distortion
#    + the two rim-bias constants, per-view 6-DoF pose, ~50k rim observations.
step "bundle adjustment" \
    "$PY" "$BP/bundle_rims.py" --init "$FIT" \
        --rim-bias b0 --out "$OUT/bundle.json"

# 4. Robot calibration: joint offsets + hand-eye. A property of YOUR arm and
#    camera mount, not of the board, so it is fitted once across many board
#    placements and then reused. Skipped unless --calibrate.
CALIB="$OUT/kinematic_calib_v3.json"
if [[ $CALIBRATE -eq 1 ]]; then
    step "robot calibration" \
        "$PY" "$BP/kinematic_calib.py" --src "$OUT/bundle.json" \
            --out "$OUT/kinematic_calib.json"
    CALIB="$OUT/kinematic_calib.json"
elif [[ ! -f "$CALIB" ]]; then
    echo "no robot calibration at $CALIB -- rerun with --calibrate" >&2
    exit 1
fi

# 5. The answer: one T_base_board from all views of this placement.
step "board pose (pose_$POSE)" \
    "$PY" "$BP/fit_board_multiview.py" --pose "$POSE" \
        --bundle "$OUT/bundle.json" --calib "$CALIB" \
        --split-half --out "$OUT/board_pose_$POSE.json"

say "DONE -> $OUT/board_pose_$POSE.json"
