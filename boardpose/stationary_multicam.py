#!/usr/bin/env python3
"""Fit a stationary board observed by one or more fixed cameras.

The output coordinate system is always the reference camera frame.  A fixed
camera rig may be supplied directly as a board-rig calibration or derived from
camera-to-base hand-eye results.  With one camera, its independently fitted
reference pose is used directly; with multiple cameras, the fixed pose of every
secondary camera is checked against the paired board poses. The board is
assumed not to move while a capture's camera frames arrive.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import fit_rims6  # noqa: E402
import seed_board  # noqa: E402
import board_model  # noqa: E402
import bundle_rims  # noqa: E402
from handeye_board import average_pose, board_corners  # noqa: E402


# Keep this local: calibrate_stationary_rig imports this module, so importing
# its version constant here would introduce a circular import.
SUPPORTED_CALIBRATION_VERSION = 7
SE3_ATOL = 1e-6
MAX_JOINT_CAMERA_MAD_PX = 1.0
MIN_HAND_EYE_SAMPLES = 5
MAX_HAND_EYE_MEAN_RMS_PX = 2.0
MAX_HAND_EYE_MAX_RMS_PX = 3.0
MAX_HAND_EYE_TRANSLATION_CROSSCHECK_M = 0.010
MAX_HAND_EYE_ROTATION_CROSSCHECK_DEG = 0.20


def _camera_intrinsics(camera_info_path):
    """Return the ROS camera calibration as OpenCV K and five-term D."""
    info = json.loads(camera_info_path.read_text())
    K = np.asarray(info["k"], dtype=float).reshape(3, 3)
    D = np.asarray(info["d"], dtype=float)
    if len(D) < 5:
        raise ValueError(f"{camera_info_path}: expected at least five distortion terms")
    return K, D[:5]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_intrinsics_provenance(camera_info_path, intrinsics):
    """Return the calibration fields used to bind a rig to a camera stream."""
    info = json.loads(camera_info_path.read_text())
    K, D = intrinsics
    return {
        "K": K.tolist(),
        "D": D.tolist(),
        "camera_info_sha256": _sha256(camera_info_path),
        "image_width": info.get("width"),
        "image_height": info.get("height"),
    }


def _capture_dirs(data_root):
    return sorted((path for path in data_root.glob("capture_*") if path.is_dir()),
                  key=lambda path: path.name)


def _validate_capture_timing(capture_dirs, maximum_span_ms=None):
    """Validate and report capture timing without requiring camera sync.

    The board is stationary while each capture is acquired, so an
    inter-camera colour timestamp span does not invalidate the observation.
    ``maximum_span_ms`` is retained for command-line compatibility and is
    recorded as a diagnostic threshold by callers; it is deliberately not a
    rejection gate.
    """
    spans = []
    for capture_dir in capture_dirs:
        metadata_path = capture_dir / "capture.json"
        try:
            span = float(json.loads(metadata_path.read_text())["camera_color_span_ms"])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{capture_dir}: missing finite camera_color_span_ms") from exc
        if not np.isfinite(span):
            raise ValueError(f"{capture_dir}: camera_color_span_ms is not finite")
        spans.append(span)
    return max(spans, default=0.0)


def _camera_names(capture_dirs):
    names = None
    for capture_dir in capture_dirs:
        present = {path.name for path in capture_dir.iterdir() if path.is_dir()
                   and (path / "color.png").exists() and (path / "depth.png").exists()}
        names = present if names is None else names & present
    return sorted(names or [])


def _set_active_camera(K, D):
    """Configure the legacy single-view fitter for one camera at a time."""
    fit_rims6.K_4K = K
    fit_rims6.D_4K = D
    seed_board.K_4K = K
    seed_board.D_4K = D


def _fit_rows(capture_dirs, cameras, intrinsics, topk, seed_span, verbose,
              out_path, data_root, restart, reference_camera, camera_poses,
              calibration_key, keep_candidates=False, camera_rim_biases=None,
              maximum_rig_disagreement_mm=4.0):
    rows = []
    dataset = str(data_root.resolve())
    if out_path.exists() and not restart:
        try:
            saved = json.loads(out_path.read_text())
            if (saved.get("dataset") == dataset and
                    saved.get("calibration_key") == calibration_key):
                rows = saved.get("view_fits", [])
                print(f"resuming: {len(rows)} image fits already in {out_path.name}")
            elif saved.get("dataset") == dataset and calibration_key is not None:
                rows = [row for row in saved.get("view_fits", [])
                        if row.get("camera") == reference_camera]
                print(f"rig calibration changed; reusing {len(rows)} independent "
                      f"{reference_camera} fits and refitting secondary cameras")
            else:
                print(f"{out_path.name} belongs to a different or older dataset; "
                      "starting fresh")
        except (OSError, ValueError, KeyError):
            print(f"could not read {out_path}; starting image fits again")
            rows = []
    if camera_poses:
        weak_secondary_rows = {
            (row["capture"], row["camera"])
            for row in rows
            if row["camera"] != reference_camera
            and row.get("mad_px", np.inf) > MAX_JOINT_CAMERA_MAD_PX
        }
        if weak_secondary_rows:
            rows = [row for row in rows
                    if (row["capture"], row["camera"]) not in weak_secondary_rows]
            print(f"retrying {len(weak_secondary_rows)} cached secondary fit(s) whose "
                  f"MAD exceeds {MAX_JOINT_CAMERA_MAD_PX:.2f} px")
    completed = {(row["capture"], row["camera"]) for row in rows}
    total_views = len(capture_dirs) * len(cameras)
    started = time.time()
    completed_this_run = 0
    for capture_dir in capture_dirs:
        for camera in cameras:
            if (capture_dir.name, camera) in completed:
                continue
            _set_active_camera(*intrinsics[camera])
            view_number = len(completed) + completed_this_run + 1
            image = cv2.imread(str(capture_dir / camera / "color.png"),
                               cv2.IMREAD_UNCHANGED)
            image_size = f"{image.shape[1]}x{image.shape[0]}" if image is not None else "unknown"
            print(f"[{view_number}/{total_views}] {capture_dir.name}/{camera}: "
                  f"starting native {image_size} fit", flush=True)

            def report(stage):
                elapsed = time.time() - started
                completed_count = len(completed) + completed_this_run
                eta = "estimating after first completed image"
                if completed_count:
                    remaining = total_views - view_number + 1
                    eta = f"ETA about {elapsed / completed_count * remaining / 60:.0f} min"
                print(f"[{view_number}/{total_views}] {capture_dir.name}/{camera}: "
                      f"{stage} ({elapsed / 60:.1f} min elapsed; {eta})", flush=True)

            reference_row = next((row for row in rows
                                  if row["capture"] == capture_dir.name
                                  and row["camera"] == reference_camera), None)
            T_reference_camera = camera_poses.get(camera)
            use_rig_prior = (camera != reference_camera and
                             reference_row is not None and
                             T_reference_camera is not None)
            if use_rig_prior:
                T_camera_board = (np.linalg.inv(T_reference_camera) @
                                  np.asarray(reference_row["T_cam_board"]))
                fitted = fit_rims6.fit_view_from_pose(capture_dir / camera,
                                                      T_camera_board,
                                                      constrain=False,
                                                      progress=report,
                                                      frozen_rim_bias=(
                                                          camera_rim_biases or {}).get(
                                                              camera))
                rejection_reason = _rig_seeded_fit_rejection(
                    reference_row, fitted, T_reference_camera,
                    maximum_rig_disagreement_mm)
                if rejection_reason is not None:
                    report(f"{rejection_reason}; retrying independent global image search")
                    fitted = fit_rims6.fit_view(
                        capture_dir / camera, topk=topk, uv_span=seed_span,
                        verbose=verbose, progress=report, bright_seed=True,
                        return_candidates=keep_candidates,
                        frozen_rim_bias=(camera_rim_biases or {}).get(camera))
                    if (fitted is not None and
                            fitted["mad_px"] > MAX_JOINT_CAMERA_MAD_PX):
                        fitted = None
            else:
                fitted = fit_rims6.fit_view(capture_dir / camera, topk=topk,
                                            uv_span=seed_span, verbose=verbose,
                                            progress=report, bright_seed=True,
                                            return_candidates=keep_candidates,
                                            frozen_rim_bias=(camera_rim_biases or {}).get(camera))
            completed_this_run += 1
            if fitted is None:
                print(f"  {capture_dir.name}/{camera:8} FAILED")
                continue
            fitted.update(capture=capture_dir.name, camera=camera)
            rows.append(fitted)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"dataset": dataset,
                                            "calibration_key": calibration_key,
                                            "view_fits": rows}, indent=2))
            print(f"  {capture_dir.name}/{camera:8} mad {fitted['mad_px']:.2f} px "
                  f"n={fitted['n']}")
    return rows


def _rig_seeded_fit_rejection(reference_row, secondary_row,
                              T_reference_secondary,
                              maximum_rig_disagreement_mm):
    """Explain why a local secondary fit needs an independent global retry."""
    if secondary_row is None:
        return "rig-seeded fit failed"
    if secondary_row["mad_px"] > MAX_JOINT_CAMERA_MAD_PX:
        return (f"rig-seeded MAD {secondary_row['mad_px']:.2f} px exceeds "
                f"{MAX_JOINT_CAMERA_MAD_PX:.2f} px")
    disagreement_mm = _fixed_rig_board_error_mm(
        np.asarray(reference_row["T_cam_board"]),
        np.asarray(secondary_row["T_cam_board"]),
        T_reference_secondary)
    if disagreement_mm > maximum_rig_disagreement_mm:
        return (f"rig-seeded pose disagrees by {disagreement_mm:.2f} mm "
                f"(limit {maximum_rig_disagreement_mm:.2f} mm)")
    return None


def _rows_by_capture_camera(rows):
    return {(row["capture"], row["camera"]): row for row in rows}


def _relative_camera_poses(rows, reference, camera):
    indexed = _rows_by_capture_camera(rows)
    transforms = []
    for capture in sorted({row["capture"] for row in rows}):
        reference_row = indexed.get((capture, reference))
        camera_row = indexed.get((capture, camera))
        if reference_row is None or camera_row is None:
            continue
        T_reference_board = np.asarray(reference_row["T_cam_board"])
        T_camera_board = np.asarray(camera_row["T_cam_board"])
        transforms.append((capture, T_reference_board @ np.linalg.inv(T_camera_board)))
    return transforms


def _corner_error_mm(transform, reference):
    delta = board_corners(transform) - board_corners(reference)
    return float(np.sqrt((delta ** 2).sum(axis=1).mean()) * 1000)


def _fixed_rig_board_error_mm(T_reference_board, T_camera_board,
                              T_reference_camera):
    """Compare local board fits at the board's observed camera-space support."""
    predicted_reference_board = T_reference_camera @ T_camera_board
    return _corner_error_mm(T_reference_board, predicted_reference_board)


def _estimate_camera_pose(relative_poses):
    if not relative_poses:
        raise RuntimeError("no synchronized fitted camera pairs are available")
    initial = average_pose([transform for _, transform in relative_poses])
    if len(relative_poses) == 1:
        return initial, np.zeros(1), None, relative_poses
    errors = np.array([_corner_error_mm(transform, initial)
                       for _, transform in relative_poses])
    median = float(np.median(errors))
    threshold = max(2.0, 3.0 * median)
    kept = [(capture, transform) for (capture, transform), error
            in zip(relative_poses, errors) if error <= threshold]
    return average_pose([transform for _, transform in kept]), errors, threshold, kept


def _intrinsic_vector(K, D):
    """Return the intrinsics layout used by bundle_rims' raw projection."""
    return np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], D]


def _pose_vector(T):
    rotation, _ = cv2.Rodrigues(np.asarray(T, float)[:3, :3])
    return np.r_[rotation.ravel(), np.asarray(T, float)[:3, 3]]


def _transform_from_pose(pose):
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(pose[:3])[0]
    transform[:3, 3] = pose[3:]
    return transform


def _joint_view(capture_dir, camera, intrinsics, T_camera_board, points):
    """Prepare raw-pixel gradients for one camera in a fixed-rig solve."""
    image = cv2.imread(str(capture_dir / camera / "color.png"), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    intrinsics_vector = _intrinsic_vector(*intrinsics[camera])
    pose = _pose_vector(T_camera_board)
    projected = bundle_rims.project(points, pose[:3], pose[3:], intrinsics_vector)
    x0, x1, y0, y1 = bundle_rims.crop_bounds(projected, image.shape)
    if x1 - x0 < 200 or y1 - y0 < 200:
        return None
    view = bundle_rims.View()
    view.name = camera
    view.label = camera
    view.pose = capture_dir.name
    view.rvec, view.tvec = pose[:3], pose[3:]
    view.gx, view.gy = bundle_rims.gradient_images(image, x0, x1, y0, y1)
    view.ox, view.oy = x0, y0
    return view


def _biased_projection(points, normals, level, pose, intrinsics, bias):
    rotation = cv2.Rodrigues(pose[:3])[0]
    model_points = board_model.biased_pts(points, normals, level, rotation,
                                          pose[3:], bias)
    return bundle_rims.project_with_normals(model_points, normals, pose[:3], pose[3:],
                                            intrinsics)


def _joint_residuals(reference_pose, views, camera_poses, observations,
                     intrinsics, residual_scales, points, normals, level,
                     camera_rim_biases):
    """Point-to-edge residuals for one board pose shared by all cameras."""
    T_reference_board = _transform_from_pose(reference_pose)
    blocks = []
    for view, observation in zip(views, observations):
        if len(observation["idx"]) == 0:
            continue
        T_camera_board = np.linalg.inv(camera_poses[view.name]) @ T_reference_board
        pose = _pose_vector(T_camera_board)
        idx = observation["idx"]
        projected, _ = _biased_projection(
            points[idx], normals[idx], level[idx], pose, intrinsics[view.name],
            camera_rim_biases[view.name])
        residual = ((projected - observation["q"]) * observation["d"]).sum(1)
        # Normalize each camera by its independently observed residual scale.
        # This prevents a noisier camera from dominating merely by contributing
        # more sampled rim points.
        blocks.append(residual / residual_scales[view.name])
    return np.concatenate(blocks) if blocks else np.zeros(1)


def _joint_fixed_rig_pose(capture_dir, rows, camera_poses, intrinsics,
                          reference_camera, camera_rim_biases=None):
    """Refine one reference-camera board pose against all raw 4K images.

    The rig transform is fixed.  Unlike the local fits, there is exactly one
    six-DoF unknown: ``T_reference_board``.  Each ICP pass re-associates rim
    edges in both *raw distorted* images before the shared-pose update.
    """
    by_camera = {row["camera"]: row for row in rows}
    reference_row = by_camera.get(reference_camera)
    if reference_row is None:
        return None
    points, normals, level, loop = board_model.build_levels()
    camera_rim_biases = camera_rim_biases or {
        camera: np.zeros(2) for camera in camera_poses}
    T_reference_board = np.asarray(reference_row["T_cam_board"], float)
    views = []
    residual_scales = {}
    for camera, T_reference_camera in camera_poses.items():
        row = by_camera.get(camera)
        if row is None:
            continue
        T_camera_board = np.linalg.inv(T_reference_camera) @ T_reference_board
        view = _joint_view(capture_dir, camera, intrinsics, T_camera_board, points)
        if view is None:
            continue
        residual_scales[camera] = max(float(row["mad_px"]), 0.10)
        views.append(view)
    if len(views) < 2:
        return None

    intrinsics_vectors = {camera: _intrinsic_vector(*values)
                          for camera, values in intrinsics.items()}
    pose = _pose_vector(T_reference_board)
    final_observations = None
    for search_halfwidth in (24, 12, 6, 4, 4):
        T_reference_board = _transform_from_pose(pose)
        for view in views:
            T_camera_board = np.linalg.inv(camera_poses[view.name]) @ T_reference_board
            camera_pose = _pose_vector(T_camera_board)
            view.rvec, view.tvec = camera_pose[:3], camera_pose[3:]
        observations = []
        for view in views:
            observations.extend(bundle_rims.measure(
                [view], intrinsics_vectors[view.name], points, normals,
                search_halfwidth, level, loop, camera_rim_biases[view.name]))
        if sum(len(observation["idx"]) for observation in observations) < 200:
            return None
        solution = least_squares(
            _joint_residuals, pose,
            args=(views, camera_poses, observations, intrinsics_vectors,
                  residual_scales, points, normals, level, camera_rim_biases),
            loss="soft_l1", f_scale=2.0,
            x_scale=[0.01, 0.01, 0.01, 0.002, 0.002, 0.002], max_nfev=80)
        if not solution.success:
            return {"joint": {"method": "fixed_rig_native_4k",
                              "status": "rejected",
                              "reason": f"optimizer did not converge: {solution.message}"}}
        if not (np.all(np.isfinite(solution.x)) and
                np.all(np.isfinite(solution.fun)) and
                np.all(np.isfinite(solution.jac))):
            return {"joint": {"method": "fixed_rig_native_4k",
                              "status": "rejected",
                              "reason": "optimizer returned non-finite values"}}
        pose = solution.x
        final_observations = observations

    T_reference_board = _transform_from_pose(pose)
    per_camera = []
    all_residuals = []
    for view, observation in zip(views, final_observations):
        T_camera_board = np.linalg.inv(camera_poses[view.name]) @ T_reference_board
        camera_pose = _pose_vector(T_camera_board)
        idx = observation["idx"]
        projected, _ = _biased_projection(
            points[idx], normals[idx], level[idx], camera_pose,
            intrinsics_vectors[view.name], camera_rim_biases[view.name])
        residual = ((projected - observation["q"]) * observation["d"]).sum(1)
        all_residuals.append(residual)
        per_camera.append({"camera": view.name, "n": int(len(residual)),
                           "rms_px": float(np.sqrt(np.mean(residual ** 2))),
                           "mad_px": float(np.median(np.abs(residual)))})
    residuals = np.concatenate(all_residuals)
    conditioning = _joint_conditioning(solution, residuals)
    if conditioning["rank"] < 6:
        return {"joint": {"method": "fixed_rig_native_4k",
                          "status": "rejected",
                          "reason": ("joint pose Jacobian is rank deficient "
                                     f"(rank {conditioning['rank']}/6)"),
                          "conditioning": conditioning}}
    excessive = [metrics for metrics in per_camera
                 if metrics["mad_px"] > MAX_JOINT_CAMERA_MAD_PX]
    if excessive:
        details = ", ".join(f"{metrics['camera']}={metrics['mad_px']:.2f} px"
                            for metrics in excessive)
        return {"joint": {"method": "fixed_rig_native_4k",
                          "status": "rejected",
                          "reason": (f"per-camera rim MAD exceeds "
                                     f"{MAX_JOINT_CAMERA_MAD_PX:.2f} px: {details}"),
                          "per_camera": per_camera,
                          "conditioning": conditioning}}
    return {"T_reference_board": T_reference_board.tolist(),
            "joint": {"method": "fixed_rig_native_4k",
                      "status": "verified",
                      "n": int(len(residuals)),
                      "rms_px": float(np.sqrt(np.mean(residuals ** 2))),
                      "mad_px": float(np.median(np.abs(residuals))),
                      "per_camera": per_camera,
                      "conditioning": conditioning}}


def _joint_conditioning(solution, residuals):
    """Report local numerical conditioning for a completed joint solve.

    These are linearized, noise-model-dependent precision proxies, not an
    accuracy claim.  The solve parameters are Rodrigues radians followed by
    metres, and the residuals are normalized by each view's local MAD.
    """
    parameter_scale = np.array([0.01, 0.01, 0.01, 0.002, 0.002, 0.002])
    jacobian = np.asarray(solution.jac, dtype=float)
    scaled_jacobian = jacobian * parameter_scale[None, :]
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(scaled_jacobian))
    if len(singular_values) and singular_values[-1] > 0:
        condition_number = float(singular_values[0] / singular_values[-1])
    else:
        condition_number = None

    degrees_of_freedom = max(len(residuals) - len(parameter_scale), 1)
    normalized_sigma = float(np.sqrt(np.sum(solution.fun ** 2) /
                                     degrees_of_freedom))
    covariance = normalized_sigma ** 2 * np.linalg.pinv(jacobian.T @ jacobian)
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        "parameter_units": ["radians", "radians", "radians", "metres",
                            "metres", "metres"],
        "parameter_scaling": {"rotation_radians": 0.01,
                              "translation_metres": 0.002},
        "rank": rank,
        "condition_number_scaled": condition_number,
        "local_precision_proxy": {
            "rotation_rms_degrees": float(np.sqrt(np.mean(
                standard_deviation[:3] ** 2)) * 180.0 / np.pi),
            "translation_rms_mm": float(np.sqrt(np.mean(
                standard_deviation[3:] ** 2)) * 1000.0),
            "meaning": ("linearized local precision proxy, not absolute "
                        "pose accuracy"),
        },
        "optimizer": {"success": bool(solution.success),
                      "optimality": float(solution.optimality),
                      "nfev": int(solution.nfev)},
    }


def _board_poses(rows, camera_poses, reference_camera=None):
    indexed = _rows_by_capture_camera(rows)
    captures = sorted({row["capture"] for row in rows})
    board_poses = []
    for capture in captures:
        if reference_camera is not None:
            reference_row = indexed.get((capture, reference_camera))
            if reference_row is not None:
                board_poses.append({"capture": capture,
                                    "T_reference_board": reference_row["T_cam_board"],
                                    "per_camera": [{"camera": reference_camera,
                                                    "corner_error_mm": 0.0}]})
                continue
        estimates = []
        for camera, T_reference_camera in camera_poses.items():
            row = indexed.get((capture, camera))
            if row is not None:
                estimates.append(T_reference_camera @ np.asarray(row["T_cam_board"]))
        if not estimates:
            continue
        T_reference_board = average_pose(estimates)
        per_camera = []
        for camera, estimate in zip([camera for camera in camera_poses
                                     if indexed.get((capture, camera)) is not None], estimates):
            per_camera.append({"camera": camera,
                               "corner_error_mm": _corner_error_mm(estimate,
                                                                     T_reference_board)})
        board_poses.append({"capture": capture,
                            "T_reference_board": T_reference_board.tolist(),
                            "per_camera": per_camera})
    return board_poses


def _single_camera_board_poses(rows, camera_poses, reference_camera):
    """Return verified board poses from one calibrated reference camera."""
    board_poses = _board_poses(rows, camera_poses, reference_camera)
    for board_pose in board_poses:
        board_pose["joint"] = _single_camera_joint_status()
    return board_poses


def _unverified_reference_poses(capture_dirs, rows, reference_camera,
                                rig_disagreement):
    """Keep the independent reference pose when the supplied rig is unsafe."""
    indexed = _rows_by_capture_camera(rows)
    board_poses = []
    for capture_dir in capture_dirs:
        reference_row = indexed.get((capture_dir.name, reference_camera))
        if reference_row is None:
            continue
        board_poses.append({
            "capture": capture_dir.name,
            "T_reference_board": reference_row["T_cam_board"],
            "joint": {
                "method": "fixed_rig_native_4k",
                "status": "unverified_calibration",
                "reason": ("supplied camera rig disagrees with independently "
                           "refined synchronized views; retaining the local "
                           f"{reference_camera} pose"),
                "rig_disagreement": rig_disagreement,
            },
            "per_camera": [{"camera": reference_camera,
                            "local_corner_error_mm": 0.0}],
        })
    return board_poses


def _provisional_joint_status():
    """Describe an in-session transform that must not be reused as a rig file."""
    return {
        "method": "in_session_camera_transform",
        "status": "provisional_in_session_estimate",
        "reason": ("camera transform was estimated from this session and is "
                   "diagnostic only; run calibrate_stationary_rig.py before "
                   "using it as a reusable rig calibration"),
    }


def _single_camera_joint_status():
    """Describe a board pose measured directly in the calibrated camera."""
    return {
        "method": "single_reference_camera",
        "status": "verified",
        "reason": ("only the independently calibrated reference camera was "
                    "used; no camera-to-camera transform was applied"),
    }


def _fixed_rig_is_verified(relative, median_error_mm, maximum_error_mm):
    """Return whether current image fits support an accepted fixed camera rig.

    A current synchronized pair is sufficient to verify a previously accepted
    rig for a one-capture measurement.  A newly estimated rig still remains
    provisional: it has no independent observation against which to check the
    camera-to-camera transform.
    """
    if not relative or median_error_mm is None:
        return False
    return median_error_mm <= maximum_error_mm


def _fixed_rig_board_poses(capture_dirs, rows, camera_poses, intrinsics,
                           reference_camera, camera_rim_biases=None):
    """Return joint fixed-rig board poses, keeping local poses as diagnostics."""
    indexed = _rows_by_capture_camera(rows)
    board_poses = []
    for capture_dir in capture_dirs:
        capture_rows = [row for row in rows if row["capture"] == capture_dir.name]
        joint_pose = _joint_fixed_rig_pose(capture_dir, capture_rows, camera_poses,
                                           intrinsics, reference_camera, camera_rim_biases)
        if joint_pose is None or joint_pose["joint"]["status"] != "verified":
            reference_row = indexed.get((capture_dir.name, reference_camera))
            if reference_row is None:
                continue
            rejection_reason = (joint_pose or {}).get("joint", {}).get("reason")
            board_poses.append({
                "capture": capture_dir.name,
                "T_reference_board": reference_row["T_cam_board"],
                "joint": {"method": "fixed_rig_native_4k", "status": "failed",
                          **({"reason": rejection_reason} if rejection_reason else {})},
                "per_camera": [],
            })
            print(f"{capture_dir.name}: joint rig refinement failed; using local "
                  f"{reference_camera} pose", flush=True)
            continue
        T_reference_board = np.asarray(joint_pose["T_reference_board"])
        per_camera = []
        for camera, T_reference_camera in camera_poses.items():
            row = indexed.get((capture_dir.name, camera))
            if row is None:
                continue
            local_reference_board = (T_reference_camera @
                                     np.asarray(row["T_cam_board"]))
            per_camera.append({
                "camera": camera,
                "local_corner_error_mm": _corner_error_mm(local_reference_board,
                                                            T_reference_board),
            })
        joint_pose["capture"] = capture_dir.name
        joint_pose["per_camera"] = per_camera
        board_poses.append(joint_pose)
        print(f"{capture_dir.name}: joint native-4K rig pose MAD "
              f"{joint_pose['joint']['mad_px']:.2f} px n={joint_pose['joint']['n']}",
              flush=True)
    return board_poses


def _validate_se3(transform, description, require_identity=False):
    """Validate a finite camera pose represented as a homogeneous SE(3) matrix."""
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{description}: expected a 4x4 transform, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{description}: transform contains non-finite values")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=SE3_ATOL, rtol=0):
        raise ValueError(f"{description}: last row must be [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=SE3_ATOL, rtol=0):
        raise ValueError(f"{description}: rotation must be orthonormal")
    determinant = np.linalg.det(rotation)
    if not np.isclose(determinant, 1.0, atol=SE3_ATOL, rtol=0):
        raise ValueError(f"{description}: rotation determinant must be +1, got {determinant}")
    if require_identity and not np.allclose(matrix, np.eye(4), atol=SE3_ATOL, rtol=0):
        raise ValueError(f"{description}: reference-camera transform must be identity")
    return matrix


def _validate_calibration_intrinsics(path, calibration, cameras, intrinsics,
                                     camera_info_paths):
    recorded = calibration.get("camera_intrinsics")
    if not isinstance(recorded, dict):
        raise ValueError(f"{path}: missing camera_intrinsics provenance")
    for camera in cameras:
        expected = recorded.get(camera)
        if not isinstance(expected, dict):
            raise ValueError(f"{path}: missing intrinsics provenance for {camera!r}")
        actual = _camera_intrinsics_provenance(camera_info_paths[camera], intrinsics[camera])
        for field in ("K", "D"):
            try:
                matches = np.allclose(np.asarray(expected[field], dtype=float),
                                      np.asarray(actual[field], dtype=float),
                                      atol=1e-9, rtol=1e-9)
            except (KeyError, TypeError, ValueError):
                matches = False
            if not matches:
                raise ValueError(f"{path}: {camera} {field} differs from the current "
                                 "camera_info.json; recalibrate this rig")
        for field in ("image_width", "image_height", "camera_info_sha256"):
            if field in expected and expected[field] != actual[field]:
                raise ValueError(f"{path}: {camera} {field} differs from the current "
                                 "camera_info.json; recalibrate this rig")


def _load_rim_biases(path, calibration, cameras):
    model = calibration.get("rim_bias_model")
    if not isinstance(model, dict) or model.get("version") != 1:
        raise ValueError(f"{path}: missing supported rim_bias_model version 1")
    per_camera = model.get("per_camera")
    if not isinstance(per_camera, dict):
        raise ValueError(f"{path}: rim_bias_model missing per_camera entries")
    biases = {}
    for camera in cameras:
        entry = per_camera.get(camera)
        if not isinstance(entry, dict) or entry.get("names") != ["b0_top", "b0_floor"]:
            raise ValueError(f"{path}: {camera} has no supported b0 rim-bias entry")
        bias = np.asarray(entry.get("bias_m"), dtype=float)
        if bias.shape != (2,) or not np.all(np.isfinite(bias)):
            raise ValueError(f"{path}: {camera} rim bias must contain two finite metre values")
        if np.any(np.abs(bias) > 0.0005):
            raise ValueError(f"{path}: {camera} rim bias exceeds the 0.5 mm model bound")
        biases[camera] = bias
    return biases


def _load_calibration(path, reference_camera, cameras, intrinsics, camera_info_paths):
    """Load a fixed camera rig and validate that it matches this capture."""
    calibration = json.loads(path.read_text())
    if calibration.get("version") != SUPPORTED_CALIBRATION_VERSION:
        raise ValueError(f"{path}: unsupported calibration version "
                         f"{calibration.get('version')!r}; expected "
                         f"{SUPPORTED_CALIBRATION_VERSION}")
    status = calibration.get("calibration_status")
    if status != "accepted":
        raise ValueError(f"{path}: calibration_status is {status!r}, not 'accepted'")
    if calibration.get("reference_camera") != reference_camera:
        raise ValueError(f"{path}: reference camera is "
                         f"{calibration.get('reference_camera')!r}, expected "
                         f"{reference_camera!r}")
    poses = calibration.get("camera_poses_in_reference")
    if not isinstance(poses, dict):
        raise ValueError(f"{path}: missing camera_poses_in_reference")
    missing = set(cameras) - set(poses)
    if missing:
        raise ValueError(f"{path}: no pose for camera(s) {sorted(missing)}")
    camera_poses = {}
    for camera in cameras:
        camera_poses[camera] = _validate_se3(
            poses[camera], f"{path}: pose for {camera!r}",
            require_identity=(camera == reference_camera))
    _validate_calibration_intrinsics(path, calibration, cameras, intrinsics,
                                     camera_info_paths)
    calibration["_camera_rim_biases"] = _load_rim_biases(path, calibration, cameras)
    return calibration, camera_poses


def _parse_camera_base_calibration_specification(value):
    """Return one ``camera=path`` command-line calibration specification."""
    try:
        camera, filename = value.split("=", 1)
    except ValueError as error:
        raise ValueError(
            "camera base calibration must use CAMERA=PATH, for example "
            "mega=/franka_ws/session_mega_calibration.json"
        ) from error
    if not camera or not filename:
        raise ValueError("camera base calibration must use a non-empty CAMERA=PATH")
    return camera, Path(filename)


def _load_camera_base_calibrations(specifications, reference_camera, cameras):
    """Derive fixed camera poses from hand-eye results in one robot base frame.

    A hand-eye result maps points from its camera frame into ``base_frame``.
    The stationary refiner needs poses that map each camera into the reference
    camera frame, so it uses ``inv(T_base_reference) @ T_base_camera``.
    """
    paths = {}
    for value in specifications:
        camera, path = _parse_camera_base_calibration_specification(value)
        if camera in paths:
            raise ValueError(f"camera base calibration for {camera!r} was given twice")
        paths[camera] = path

    expected = set(cameras)
    if set(paths) != expected:
        missing = sorted(expected - set(paths))
        unexpected = sorted(set(paths) - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError("camera base calibrations must match captured cameras: "
                         + "; ".join(details))

    transforms = {}
    base_frame = None
    source_dataset = None
    provenance = {}
    for camera in cameras:
        path = paths[camera]
        try:
            payload = json.loads(path.read_text())
        except OSError as error:
            raise ValueError(f"could not read {path}: {error}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} is not valid JSON: {error}") from error

        schema_version = payload.get("schema_version")
        if schema_version not in (1, 2):
            raise ValueError(f"{path}: expected hand-eye schema_version 1 or joint schema 2")
        if payload.get("units") != "m":
            raise ValueError(f"{path}: expected hand-eye units 'm'")

        candidate_base_frame = payload.get("base_frame")
        if not isinstance(candidate_base_frame, str) or not candidate_base_frame:
            raise ValueError(f"{path}: missing non-empty base_frame")
        if schema_version == 1:
            expected_camera_frame = f"{camera}_camera"
            if payload.get("camera_frame") != expected_camera_frame:
                raise ValueError(f"{path}: camera_frame must be {expected_camera_frame!r}")
        else:
            if payload.get("calibration_type") != "joint_robot_camera_handeye":
                raise ValueError(f"{path}: schema 2 is not a joint hand-eye artifact")
            frames = payload.get("camera_frames")
            transforms_by_camera = payload.get("T_base_camera")
            if not isinstance(frames, dict) or not isinstance(transforms_by_camera, dict):
                raise ValueError(f"{path}: joint artifact is missing camera frames or transforms")
            expected_camera_frame = frames.get(camera)
            if not isinstance(expected_camera_frame, str) or not expected_camera_frame:
                raise ValueError(f"{path}: joint artifact has no frame for {camera!r}")
        if base_frame is None:
            base_frame = candidate_base_frame
        elif candidate_base_frame != base_frame:
            raise ValueError(f"{path}: base_frame {candidate_base_frame!r} differs from "
                             f"{base_frame!r}")

        handeye_provenance = payload.get("provenance")
        if not isinstance(handeye_provenance, dict):
            raise ValueError(f"{path}: missing hand-eye provenance")
        if schema_version == 1 and handeye_provenance.get("camera_name") != camera:
            raise ValueError(f"{path}: provenance.camera_name must be {camera!r}")
        if schema_version == 2 and camera not in handeye_provenance.get("camera_names", []):
            raise ValueError(f"{path}: joint provenance does not contain {camera!r}")
        candidate_dataset = handeye_provenance.get("dataset")
        if not isinstance(candidate_dataset, str) or not candidate_dataset:
            raise ValueError(f"{path}: provenance.dataset must identify the taught capture")
        if source_dataset is None:
            source_dataset = candidate_dataset
        elif candidate_dataset != source_dataset:
            raise ValueError(
                "camera base calibrations must come from the same taught-capture "
                f"dataset (got {candidate_dataset!r} and {source_dataset!r})"
            )

        quality = payload.get("quality")
        if not isinstance(quality, dict):
            raise ValueError(f"{path}: missing hand-eye quality metrics")
        if schema_version == 2:
            quality_by_camera = quality.get("per_camera")
            if not isinstance(quality_by_camera, dict):
                raise ValueError(f"{path}: joint quality is missing per-camera metrics")
            per_camera_quality = quality_by_camera.get(camera)
            if not isinstance(per_camera_quality, dict):
                raise ValueError(f"{path}: joint quality has no entry for {camera!r}")
            samples_used = per_camera_quality.get("samples_used")
            mean_reprojection = per_camera_quality.get(
                "direct_corner_rms_px",
                per_camera_quality.get("mean_reprojection_rms_px"))
            max_reprojection = per_camera_quality.get(
                "direct_corner_max_rms_px",
                per_camera_quality.get(
                    "direct_corner_max_px",
                    per_camera_quality.get("max_reprojection_rms_px")))
            mean_metric_name = ("direct_corner_rms_px"
                                if "direct_corner_rms_px" in per_camera_quality
                                else "mean_reprojection_rms_px")
            max_metric_name = ("direct_corner_max_rms_px"
                               if "direct_corner_max_rms_px" in per_camera_quality
                               else ("direct_corner_max_px"
                                     if "direct_corner_max_px" in per_camera_quality
                                     else "max_reprojection_rms_px"))
            quality_for_provenance = per_camera_quality | {
                "joint_samples_total": quality.get("samples_total"),
                "joint_samples_used": quality.get("samples_used"),
            }
            validation = payload.get("validation")
            if not isinstance(validation, dict):
                raise ValueError(f"{path}: joint calibration has no validation report")
            if validation.get("status") != "completed":
                raise ValueError(
                    f"{path}: joint calibration validation status is "
                    f"{validation.get('status', 'missing')!r}")
            if validation.get("method") != "leave_one_pose_out":
                raise ValueError(f"{path}: unsupported joint validation method")
            if validation.get("observation_model") != "direct_distorted_image_corners":
                raise ValueError(f"{path}: joint validation is not based on direct image corners")
            if validation.get("failures"):
                raise ValueError(f"{path}: joint calibration validation has failed folds")
        else:
            samples_used = quality.get("samples_used")
            mean_reprojection = quality.get("mean_reprojection_rms_px")
            max_reprojection = quality.get("max_reprojection_rms_px")
            quality_for_provenance = quality
        if not isinstance(samples_used, int) or samples_used < MIN_HAND_EYE_SAMPLES:
            raise ValueError(f"{path}: hand-eye calibration needs at least "
                             f"{MIN_HAND_EYE_SAMPLES} samples")
        metrics = [
            (mean_metric_name if schema_version == 2 else "mean_reprojection_rms_px",
             mean_reprojection,
             MAX_HAND_EYE_MEAN_RMS_PX),
            (max_metric_name if schema_version == 2 else "max_reprojection_rms_px",
             max_reprojection,
             MAX_HAND_EYE_MAX_RMS_PX),
        ]
        if schema_version == 1:
            metrics.extend([
                ("park_andreff_translation_disagreement_m",
                 quality.get("park_andreff_translation_disagreement_m"),
                 MAX_HAND_EYE_TRANSLATION_CROSSCHECK_M),
                ("park_andreff_rotation_disagreement_deg",
                 quality.get("park_andreff_rotation_disagreement_deg"),
                 MAX_HAND_EYE_ROTATION_CROSSCHECK_DEG),
            ])
        for metric, value, limit in metrics:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{path}: quality.{metric} must be finite")
            if value < 0:
                raise ValueError(f"{path}: quality.{metric} must be non-negative")
            if value > limit:
                print(f"diagnostic: {path}: quality.{metric}={value:g} exceeds "
                      f"the {limit:g} reference limit; continuing", flush=True)

        transform_value = (payload.get("T_base_camera") if schema_version == 1
                           else payload["T_base_camera"].get(camera))
        transforms[camera] = _validate_se3(
            transform_value, f"{path}: T_base_camera for {camera!r}")
        provenance[camera] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "camera_frame": expected_camera_frame,
            "base_frame": candidate_base_frame,
            "dataset": candidate_dataset,
            "quality": quality_for_provenance,
            "calibration_schema_version": schema_version,
        }

    T_base_reference = transforms[reference_camera]
    poses = {reference_camera: np.eye(4)}
    for camera in cameras:
        if camera != reference_camera:
            poses[camera] = np.linalg.inv(T_base_reference) @ transforms[camera]
    return poses, provenance


def _camera_base_calibration_key(provenance):
    """Return a cache key that changes with either hand-eye source file."""
    canonical = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    return f"handeye:{hashlib.sha256(canonical.encode()).hexdigest()}"


def _calibration_key(path):
    """A changed rig file must invalidate pose fits made with its predecessor."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"{path.resolve()}:{digest}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "run_01")
    parser.add_argument("--reference-camera", default="mega")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed-span", type=float, default=0.55,
                        help="initial board-position search radius in metres")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--restart", action="store_true",
                        help="discard completed per-image fits instead of resuming")
    parser.add_argument("--calibration", type=Path,
                        help="fixed camera-rig calibration JSON; constrains secondary views")
    parser.add_argument("--camera-base-calibration", action="append", default=[],
                        metavar="CAMERA=PATH",
                        help=("hand-eye JSON for every captured camera; derives the fixed "
                              "rig from their common robot base frame"))
    parser.add_argument("--optical-calibration", type=Path,
                        help=("optional accepted board-rig JSON used only for its validated "
                              "per-camera rim-bias terms; camera poses are ignored"))
    parser.add_argument("--max-calibrated-rig-disagreement-mm", type=float, default=4.0,
                        help="mark a calibrated result unverified above this image-to-rig disagreement")
    parser.add_argument("--max-camera-color-span-ms", type=float, default=75.0,
                        help=("report spans above this diagnostic value; do not reject "
                              "them because the board is stationary"))
    parser.add_argument("--out", type=Path,
                        default=HERE / "out" / "stationary_multicam.json")
    args = parser.parse_args()

    if args.calibration is not None and args.camera_base_calibration:
        raise SystemExit("use either --calibration or --camera-base-calibration, not both")
    if args.optical_calibration is not None and not args.camera_base_calibration:
        raise SystemExit("--optical-calibration requires --camera-base-calibration")

    capture_dirs = _capture_dirs(args.data)
    cameras = _camera_names(capture_dirs)
    if not capture_dirs:
        raise SystemExit(f"no capture_* directories in {args.data}")
    if args.reference_camera not in cameras:
        raise SystemExit(f"reference camera {args.reference_camera!r} not in {cameras}")
    try:
        maximum_observed_span_ms = _validate_capture_timing(
            capture_dirs, args.max_camera_color_span_ms)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    camera_info_paths = {camera: capture_dirs[0] / camera / "camera_info.json"
                         for camera in cameras}
    intrinsics = {camera: _camera_intrinsics(camera_info_paths[camera])
                  for camera in cameras}
    calibration = None
    fixed_camera_poses = None
    calibration_key = None
    rig_source = None
    if args.calibration is not None:
        try:
            calibration, fixed_camera_poses = _load_calibration(
                args.calibration, args.reference_camera, cameras,
                intrinsics, camera_info_paths)
        except (OSError, ValueError) as error:
            raise SystemExit(error) from error
        calibration_key = _calibration_key(args.calibration)
        cameras = [args.reference_camera] + [camera for camera in cameras
                                             if camera != args.reference_camera]
        print(f"using fixed camera rig from {args.calibration}", flush=True)
        rig_source = {"mode": "board_rig_calibration",
                      "path": str(args.calibration.resolve())}
    elif args.camera_base_calibration:
        try:
            fixed_camera_poses, handeye_provenance = _load_camera_base_calibrations(
                args.camera_base_calibration, args.reference_camera, cameras)
        except ValueError as error:
            raise SystemExit(error) from error
        rim_bias_source = None
        if args.optical_calibration is not None:
            try:
                optical_calibration, _ = _load_calibration(
                    args.optical_calibration, args.reference_camera, cameras,
                    intrinsics, camera_info_paths)
            except (OSError, ValueError) as error:
                raise SystemExit(error) from error
            rim_biases = optical_calibration["_camera_rim_biases"]
            rim_bias_source = {
                "path": str(args.optical_calibration.resolve()),
                "sha256": _sha256(args.optical_calibration),
            }
            print(f"reusing validated rim-bias terms from {args.optical_calibration}; "
                  "its camera poses are ignored", flush=True)
        else:
            rim_biases = {camera: np.zeros(2) for camera in cameras}
        calibration = {"_camera_rim_biases": rim_biases}
        calibration_key_payload = {"handeye": handeye_provenance}
        if rim_bias_source is not None:
            calibration_key_payload["optical"] = rim_bias_source
        calibration_key = _camera_base_calibration_key(calibration_key_payload)
        cameras = [args.reference_camera] + [camera for camera in cameras
                                             if camera != args.reference_camera]
        if rim_bias_source is None:
            rim_bias_model = "zero_constant_offsets"
        else:
            rim_bias_model = "validated_board_optical_prior"
        rig_source = {"mode": "derived_from_camera_base_calibrations",
                      "base_frame": handeye_provenance[args.reference_camera]["base_frame"],
                      "camera_base_calibrations": handeye_provenance,
                      "rim_bias_model": rim_bias_model}
        if rim_bias_source is not None:
            rig_source["optical_calibration"] = rim_bias_source
        print("using fixed camera rig derived from camera-to-base hand-eye results",
              flush=True)
    if maximum_observed_span_ms > args.max_camera_color_span_ms:
        print(f"notice: maximum inter-camera colour span is "
              f"{maximum_observed_span_ms:.1f} ms, above the diagnostic "
              f"value {args.max_camera_color_span_ms:.1f} ms; "
              "the stationary-board pipeline accepts this", flush=True)
    print(f"{len(capture_dirs)} stationary multi-camera captures; "
          f"cameras: {', '.join(cameras)}")
    started = time.time()
    rows = _fit_rows(capture_dirs, cameras, intrinsics, args.topk,
                     args.seed_span, args.verbose, args.out, args.data,
                     args.restart, args.reference_camera,
                     fixed_camera_poses or {},
                     calibration_key,
                     camera_rim_biases=(calibration or {}).get("_camera_rim_biases"),
                     maximum_rig_disagreement_mm=(
                         args.max_calibrated_rig_disagreement_mm))
    if not rows:
        raise SystemExit("no views fitted")

    print("all image fits complete; estimating fixed camera-to-camera transforms", flush=True)
    camera_poses = fixed_camera_poses or {args.reference_camera: np.eye(4)}
    camera_diagnostics = {}
    for camera in cameras:
        if camera == args.reference_camera:
            continue
        relative = _relative_camera_poses(rows, args.reference_camera, camera)
        if fixed_camera_poses is None:
            T_reference_camera, errors, threshold, kept = _estimate_camera_pose(relative)
            camera_poses[camera] = T_reference_camera
        else:
            T_reference_camera = camera_poses[camera]
            indexed = _rows_by_capture_camera(rows)
            errors = np.array([
                _fixed_rig_board_error_mm(
                    np.asarray(indexed[(capture, args.reference_camera)]["T_cam_board"]),
                    np.asarray(indexed[(capture, camera)]["T_cam_board"]),
                    T_reference_camera)
                for capture, _ in relative
            ])
            threshold = None
            kept = relative
        median_error = float(np.median(errors)) if len(errors) else None
        is_verified = False
        status = "provisional_in_session_estimate"
        if fixed_camera_poses is not None:
            status = "verified"
            is_verified = _fixed_rig_is_verified(
                relative, median_error, args.max_calibrated_rig_disagreement_mm)
            if not is_verified:
                status = "unverified_calibration"
        camera_diagnostics[camera] = {
            "pair_corner_errors_mm": {capture: float(error)
                                      for (capture, _), error in zip(relative, errors)},
            "median_corner_error_mm": median_error,
            "outlier_threshold_mm": threshold,
            "captures_used": [capture for capture, _ in kept],
            "is_verified": is_verified,
            "status": status,
        }
        if len(relative) == 1 and fixed_camera_poses is None:
            print(f"{args.reference_camera} <- {camera}: estimated from one pair only; "
                  "not independently verified", flush=True)
        elif len(relative) == 1:
            print(f"{args.reference_camera} <- {camera}: one paired view; "
                  f"disagreement {median_error:.2f} mm", flush=True)
        elif relative:
            print(f"{args.reference_camera} <- {camera}: {len(kept)}/{len(relative)} pairs "
                  f"used; median disagreement {median_error:.2f} mm", flush=True)
        else:
            print(f"{args.reference_camera} <- {camera}: no paired fitted view",
                  flush=True)
        if fixed_camera_poses is not None and not is_verified:
            disagreement = ("unavailable" if median_error is None
                            else f"{median_error:.2f} mm")
            print(f"WARNING: calibrated {args.reference_camera} <- {camera} disagrees "
                  f"with the local image refinements by {disagreement} "
                  f"(limit {args.max_calibrated_rig_disagreement_mm:.2f} mm). "
                  "Do not use this rig pose until the camera mount and calibration are checked.",
                  flush=True)

    if len(cameras) == 1:
        # A hand-eye file is still required to move the result into the robot
        # base, but there is no camera-to-camera rig to verify or fuse.  The
        # reference image fit is already the complete stationary measurement.
        print("one camera available; using its calibrated image pose directly",
              flush=True)
        board_poses = _single_camera_board_poses(
            rows, camera_poses, args.reference_camera)
    elif fixed_camera_poses is not None:
        unverified_cameras = [camera for camera, diagnostic
                              in camera_diagnostics.items()
                              if not diagnostic["is_verified"]]
        if unverified_cameras:
            rig_disagreement = {
                camera: {
                    "median_corner_error_mm": camera_diagnostics[camera]["median_corner_error_mm"],
                    "limit_mm": args.max_calibrated_rig_disagreement_mm,
                    "pair_corner_errors_mm": camera_diagnostics[camera]["pair_corner_errors_mm"],
                }
                for camera in unverified_cameras
            }
            print("rig calibration is unverified; not running shared-pose fusion. "
                  f"Publishing independent {args.reference_camera} image poses.",
                  flush=True)
            board_poses = _unverified_reference_poses(capture_dirs, rows,
                                                       args.reference_camera,
                                                       rig_disagreement)
        else:
            print("rig calibration verified; refining one shared board pose per "
                  "capture against all raw 4K views", flush=True)
            board_poses = _fixed_rig_board_poses(capture_dirs, rows, camera_poses,
                                                  intrinsics, args.reference_camera,
                                                  calibration["_camera_rim_biases"])
    else:
        if len(cameras) == 1:
            board_poses = _single_camera_board_poses(
                rows, camera_poses, args.reference_camera)
        else:
            board_poses = _board_poses(rows, camera_poses, args.reference_camera)
            for board_pose in board_poses:
                board_pose["joint"] = _provisional_joint_status()

    output = {
        "dataset": str(args.data.resolve()),
        "frame": f"{args.reference_camera}_camera",
        "reference_camera": args.reference_camera,
        "camera_intrinsics": {
            camera: {"K": K.tolist(), "D": D.tolist()} for camera, (K, D) in intrinsics.items()
        },
        "camera_poses_in_reference": {camera: pose.tolist()
                                        for camera, pose in camera_poses.items()},
        "camera_diagnostics": camera_diagnostics,
        "calibration": (str(args.calibration.resolve()) if args.calibration is not None else None),
        "calibration_key": calibration_key,
        "rig_source": rig_source,
        "capture_timing": {
            "policy": "diagnostic_only_stationary_board",
            "diagnostic_camera_color_span_threshold_ms": args.max_camera_color_span_ms,
            "maximum_observed_camera_color_span_ms": maximum_observed_span_ms,
        },
        "view_fits": rows,
        "board_poses": board_poses,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(f"wrote {args.out} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
