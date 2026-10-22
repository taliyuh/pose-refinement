#!/usr/bin/env python3
"""Compose a session camera calibration and a stationary board fit.

The result is the small, versioned contract consumed by the robot manipulation
node.  All transforms use metres and map a point from the named child frame
into the named parent frame:

    p_base = T_base_board @ p_board
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_BOARD_SURFACE_Z_M = 0.025


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except OSError as error:
        raise ValueError(f"could not read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error


def _transform(value, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains a non-finite value")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")

    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    determinant = np.linalg.det(rotation)
    if not np.isclose(determinant, 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant is {determinant:.8f}, expected +1")
    return matrix


def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calibration_transform(calibration: dict, camera: str | None = None) -> tuple[str, str, np.ndarray]:
    try:
        base_frame = calibration["base_frame"]
        if calibration.get("schema_version") == 2:
            transforms = calibration["T_base_camera"]
            frames = calibration["camera_frames"]
            if not isinstance(transforms, dict) or not isinstance(frames, dict):
                raise ValueError("joint calibration camera transforms and frames must be mappings")
            if camera is None:
                if len(transforms) != 1:
                    raise ValueError(
                        "joint calibration contains multiple cameras; pass --camera"
                    )
                camera = next(iter(transforms))
            camera_frame = frames[camera]
            matrix = transforms[camera]
        else:
            camera_frame = calibration["camera_frame"]
            matrix = calibration["T_base_camera"]
    except KeyError as error:
        raise ValueError(
            "calibration must contain base_frame, camera_frame, and T_base_camera"
        ) from error
    if not isinstance(base_frame, str) or not base_frame:
        raise ValueError("calibration base_frame must be a non-empty string")
    if not isinstance(camera_frame, str) or not camera_frame:
        raise ValueError("calibration camera_frame must be a non-empty string")
    return base_frame, camera_frame, _transform(matrix, "calibration T_base_camera")


def _board_fit(refinement: dict, capture: str | None) -> tuple[str, str, np.ndarray, dict]:
    reference_frame = refinement.get("frame")
    if not isinstance(reference_frame, str) or not reference_frame:
        raise ValueError("refinement must contain a non-empty top-level frame")

    board_poses = refinement.get("board_poses")
    if not isinstance(board_poses, list) or not board_poses:
        raise ValueError("refinement has no board_poses; run the stationary pipeline first")

    if capture is None:
        if len(board_poses) != 1:
            names = ", ".join(str(row.get("capture", "?")) for row in board_poses)
            raise ValueError(f"refinement has {len(board_poses)} captures ({names}); pass --capture")
        selected = board_poses[0]
    else:
        selected = next((row for row in board_poses if row.get("capture") == capture), None)
        if selected is None:
            available = ", ".join(str(row.get("capture", "?")) for row in board_poses)
            raise ValueError(f"capture {capture!r} is absent; available: {available}")

    joint_status = selected.get("joint", {}).get("status")
    if joint_status != "verified":
        raise ValueError(
            f"capture {selected.get('capture', '?')} has joint status {joint_status!r}; "
            "only a verified stationary refinement may be exported for robot use"
        )

    matrix = selected.get("T_reference_board")
    if matrix is None:
        raise ValueError(
            f"capture {selected.get('capture', '?')} has no T_reference_board; "
            "it was not successfully refined"
        )
    return reference_frame, str(selected["capture"]), _transform(
        matrix, "refinement T_reference_board"
    ), selected


def compose(calibration: dict, refinement: dict, capture: str | None,
            camera: str | None = None) -> dict:
    """Return the canonical base-to-board artifact for one stationary capture."""
    base_frame, camera_frame, T_base_camera = _calibration_transform(calibration, camera)
    reference_frame, capture_name, T_camera_board, selected = _board_fit(refinement, capture)
    if camera_frame != reference_frame:
        raise ValueError(
            f"frame mismatch: calibration camera_frame is {camera_frame!r}, but "
            f"refinement frame is {reference_frame!r}"
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "units": "m",
        "base_frame": base_frame,
        "board_frame": "board",
        "board_surface_z_m": DEFAULT_BOARD_SURFACE_Z_M,
        "T_base_board": (T_base_camera @ T_camera_board).tolist(),
        "source": {
            "camera_frame": camera_frame,
            "capture": capture_name,
            "refinement_joint": selected.get("joint"),
        },
    }
    return result


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True,
                        help="session base-to-reference-camera calibration JSON")
    parser.add_argument("--refinement", type=Path, required=True,
                        help="stationary_multicam JSON")
    parser.add_argument("--capture", help="capture name when refinement has multiple poses")
    parser.add_argument("--camera", help="camera to select from a joint calibration artifact")
    parser.add_argument("--output", type=Path, required=True,
                        help="canonical base-to-board JSON to write")
    args = parser.parse_args(argv)

    try:
        calibration = _load_json(args.calibration)
        refinement = _load_json(args.refinement)
        output = compose(calibration, refinement, args.capture, args.camera)
        output["source"].update({
            "calibration_path": str(args.calibration.resolve()),
            "calibration_sha256": _source_digest(args.calibration),
            "refinement_path": str(args.refinement.resolve()),
            "refinement_sha256": _source_digest(args.refinement),
        })
        _write_json(args.output, output)
    except ValueError as error:
        parser.error(str(error))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
