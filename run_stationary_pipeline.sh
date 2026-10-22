#!/usr/bin/env bash
# Fit board poses from one or more views of stationary cameras.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
DATA="$ROOT/run_01"
REFERENCE="mega"
CALIBRATION=""
CAMERA_BASE_CALIBRATIONS=()
OPTICAL_CALIBRATION=""
MAX_CAMERA_COLOR_SPAN_MS="75"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data) DATA="$2"; shift 2 ;;
        --reference-camera) REFERENCE="$2"; shift 2 ;;
        --calibration) CALIBRATION="$2"; shift 2 ;;
        --camera-base-calibration) CAMERA_BASE_CALIBRATIONS+=("$2"); shift 2 ;;
        --optical-calibration) OPTICAL_CALIBRATION="$2"; shift 2 ;;
        --max-camera-color-span-ms) MAX_CAMERA_COLOR_SPAN_MS="$2"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--data DIR] [--reference-camera NAME] [--calibration FILE] [--camera-base-calibration CAMERA=FILE] [--optical-calibration FILE] [--max-camera-color-span-ms MS]"
            exit 0
            ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

DATA_NAME="$(basename "${DATA%/}")"
OUT="$ROOT/boardpose/out/stationary_multicam_${DATA_NAME}.json"
OVERLAYS="$ROOT/boardpose/out_stationary_overlays/$DATA_NAME"

COMMAND=("$PY" "$ROOT/boardpose/stationary_multicam.py" --data "$DATA"
         --reference-camera "$REFERENCE" --out "$OUT"
         --max-camera-color-span-ms "$MAX_CAMERA_COLOR_SPAN_MS")
if [[ -n "$CALIBRATION" ]]; then
    COMMAND+=(--calibration "$CALIBRATION")
fi
for CAMERA_BASE_CALIBRATION in "${CAMERA_BASE_CALIBRATIONS[@]}"; do
    COMMAND+=(--camera-base-calibration "$CAMERA_BASE_CALIBRATION")
done
if [[ -n "$OPTICAL_CALIBRATION" ]]; then
    COMMAND+=(--optical-calibration "$OPTICAL_CALIBRATION")
fi
"${COMMAND[@]}"
echo "[stationary pipeline] rendering reprojection overlays"
"$PY" "$ROOT/boardpose/render_stationary_fit.py" --data "$DATA" \
    --src "$OUT" --outdir "$OVERLAYS"
echo "[stationary pipeline] rendering 3D rig scene"
"$PY" "$ROOT/boardpose/render_stationary_scene.py" --src "$OUT" \
    --out "$OVERLAYS/scene_3d.png"
