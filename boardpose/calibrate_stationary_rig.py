#!/usr/bin/env python3
"""Calibrate a fixed stationary camera rig from paired board captures.

The saved transform maps every secondary camera frame into the selected
reference-camera frame. Camera intrinsics are recorded but never adjusted.
"""

import argparse
import hashlib
import json
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

import stationary_multicam as stationary  # noqa: E402
from handeye_board import average_pose, board_corners  # noqa: E402
import outline_board  # noqa: E402
import board_model  # noqa: E402
import fit_rims6  # noqa: E402
import bundle_rims  # noqa: E402


CALIBRATION_VERSION = stationary.SUPPORTED_CALIBRATION_VERSION
RIM_FIT_VERSION = "independent_local_pose_v4_per_camera_b0_bias"
FROZEN_CORRESPONDENCE_MAX_SHIFT_PX = 2.0
RIM_BIAS_BOUND_M = 0.0005
RIM_BIAS_PRIOR_SIGMA_M = 0.00025


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corner_error_mm(transform, reference):
    difference = board_corners(transform) - board_corners(reference)
    return float(np.sqrt((difference ** 2).sum(axis=1).mean()) * 1000)


def _medoid_pose(relative_poses):
    """Return the pose with the smallest median board-corner distance."""
    transforms = [transform for _, transform in relative_poses]
    distances = np.empty((len(transforms), len(transforms)))
    for i, transform in enumerate(transforms):
        for j, other in enumerate(transforms):
            distances[i, j] = _corner_error_mm(transform, other)
    return transforms[int(np.argmin(np.median(distances, axis=1)))]


def _transform_clusters(relative_poses, maximum_distance_mm=4.0):
    """Group locally fitted rig hypotheses that agree at board corners.

    Each input has already passed its image-space rim-fit quality gate.  A
    disconnected group is therefore evidence for more than one physical rig
    state or a repeatable wrong local fit, not an outlier to average away.
    """
    count = len(relative_poses)
    neighbours = [[] for _ in range(count)]
    distances = np.zeros((count, count), dtype=float)
    for left, (_, transform) in enumerate(relative_poses):
        for right in range(left + 1, count):
            distance = _corner_error_mm(transform, relative_poses[right][1])
            distances[left, right] = distances[right, left] = distance
            if distance <= maximum_distance_mm:
                neighbours[left].append(right)
                neighbours[right].append(left)

    clusters = []
    unseen = set(range(count))
    while unseen:
        pending = [unseen.pop()]
        members = []
        while pending:
            index = pending.pop()
            members.append(index)
            newly_connected = [other for other in neighbours[index]
                               if other in unseen]
            for other in newly_connected:
                unseen.remove(other)
                pending.append(other)
        clusters.append(sorted(members))
    clusters.sort(key=lambda members: members[0])
    return [[relative_poses[index][0] for index in members] for members in clusters], distances


def _multiple_rig_states_error(relative_poses, maximum_distance_mm=4.0):
    """Return an actionable error when accepted local fits form separate states."""
    clusters, distances = _transform_clusters(relative_poses, maximum_distance_mm)
    if len(clusters) <= 1:
        return None, clusters, distances
    named_clusters = "; ".join(
        f"group {index + 1}: {', '.join(captures)}"
        for index, captures in enumerate(clusters))
    between_groups = []
    for left, left_captures in enumerate(clusters):
        left_indices = [index for index, (capture, _) in enumerate(relative_poses)
                        if capture in left_captures]
        for right in range(left + 1, len(clusters)):
            right_indices = [index for index, (capture, _) in enumerate(relative_poses)
                             if capture in clusters[right]]
            closest = min(distances[i, j] for i in left_indices for j in right_indices)
            between_groups.append(f"groups {left + 1}/{right + 1}: {closest:.2f} mm")
    return (f"accepted local rim fits indicate multiple camera-rig states "
            f"(agreement limit {maximum_distance_mm:.2f} mm): {named_clusters}; "
            f"closest cross-group distances: {', '.join(between_groups)}. "
            "Check these capture pairs visually, then recalibrate using captures "
            "from one unchanged camera setup."), clusters, distances


def _capture_pose_diversity(relative_poses, reference_rows):
    """Summarize the observable motion of a board constrained to the table."""
    poses = [np.asarray(reference_rows[capture]["T_cam_board"], float)
             for capture, _ in relative_poses]
    if not poses:
        return {"capture_count": 0, "warnings": []}
    normals = np.asarray([pose[:3, 2] for pose in poses])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    dot_products = np.clip(normals @ normals.T, -1.0, 1.0)
    normal_span_deg = float(np.degrees(np.arccos(dot_products.min())))

    table_normal = normals.mean(axis=0)
    table_normal /= np.linalg.norm(table_normal)
    board_x_axes = []
    for pose in poses:
        board_x = pose[:3, 0] - table_normal * np.dot(pose[:3, 0], table_normal)
        board_x_axes.append(board_x / np.linalg.norm(board_x))
    yaw_span_deg = max(
        float(np.degrees(np.arccos(np.clip(abs(np.dot(left, right)), -1.0, 1.0))))
        for left in board_x_axes
        for right in board_x_axes
    )

    centers = np.asarray([pose[:3, 3] for pose in poses])
    center_span_m = 0.0
    for left in centers:
        for right in centers:
            displacement = left - right
            planar = displacement - table_normal * np.dot(displacement, table_normal)
            center_span_m = max(center_span_m, float(np.linalg.norm(planar)))

    depths_m = np.asarray([pose[2, 3] for pose in poses])
    depth_span_mm = float((depths_m.max() - depths_m.min()) * 1000.0)
    warnings = []
    if yaw_span_deg < 45.0:
        warnings.append("flat-board yaw span is below 45 degrees")
    if center_span_m < 0.100:
        warnings.append("flat-board centre movement is below 100 mm")
    return {
        "capture_count": len(poses),
        "normal_span_deg": normal_span_deg,
        "depth_span_mm": depth_span_mm,
        "flat_board_yaw_span_deg": yaw_span_deg,
        "flat_board_center_span_mm": center_span_m * 1000.0,
        "warnings": warnings,
    }


def estimate_transform(relative_poses, minimum_pairs):
    """Robustly average board-derived camera transforms from synchronized pairs."""
    if len(relative_poses) < minimum_pairs:
        raise RuntimeError(
            f"need at least {minimum_pairs} synchronized accepted pairs; "
            f"found {len(relative_poses)}")

    medoid = _medoid_pose(relative_poses)
    initial_errors = np.array([_corner_error_mm(transform, medoid)
                               for _, transform in relative_poses])
    median = float(np.median(initial_errors))
    mad = float(np.median(np.abs(initial_errors - median)))
    threshold = max(2.0, median + 3.0 * max(mad, 0.25))
    kept = [(capture, transform) for (capture, transform), error
            in zip(relative_poses, initial_errors) if error <= threshold]
    if len(kept) < minimum_pairs:
        raise RuntimeError(
            f"only {len(kept)} pairs agree within {threshold:.2f} mm; "
            f"need {minimum_pairs}")

    transform = average_pose([pose for _, pose in kept])
    final_errors = np.array([_corner_error_mm(pose, transform)
                             for _, pose in relative_poses])
    return transform, initial_errors, final_errors, threshold, kept


def _intrinsics_provenance(capture_dir, cameras, intrinsics):
    provenance = {}
    for camera in cameras:
        info_path = capture_dir / camera / "camera_info.json"
        info = json.loads(info_path.read_text())
        provenance[camera] = {
            "camera_info": str(info_path.resolve()),
            "camera_info_sha256": _sha256(info_path),
            "K": intrinsics[camera][0].tolist(),
            "D": intrinsics[camera][1].tolist(),
            "image_width": info.get("width"),
            "image_height": info.get("height"),
        }
    return provenance


def _load_transform_recalibration_prior(path, reference_camera, cameras, intrinsics,
                                        camera_info_paths):
    """Load a prior rig only when it exactly binds to these camera streams."""
    raw = json.loads(path.read_text())
    expected_cameras = set(cameras)
    for field in ("camera_poses_in_reference", "camera_intrinsics"):
        entries = raw.get(field)
        if not isinstance(entries, dict) or set(entries) != expected_cameras:
            actual_cameras = sorted(entries) if isinstance(entries, dict) else None
            raise ValueError(f"{path}: {field} cameras {actual_cameras!r} do not exactly "
                             f"match current cameras {sorted(expected_cameras)!r}")
    calibration, poses = stationary._load_calibration(
        path, reference_camera, cameras, intrinsics, camera_info_paths)
    biases = calibration["_camera_rim_biases"]
    return calibration, poses, biases, {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "reused_rim_biases_m": {name: biases[name].tolist() for name in cameras},
    }


def _fast_recalibration_gates(diagnostics):
    """Return the first safety gate violated by transform-only recalibration."""
    local = diagnostics.get("local_transform_diagnostics", {})
    diversity = local.get("capture_pose_diversity", {})
    yaw_span = diversity.get("flat_board_yaw_span_deg")
    if yaw_span is None or yaw_span < 45.0:
        return f"flat-board yaw span is {yaw_span!r} deg (minimum 45.0 deg)"
    center_span = diversity.get("flat_board_center_span_mm")
    if center_span is None or center_span < 100.0:
        return f"flat-board centre movement is {center_span!r} mm (minimum 100 mm)"
    median = local.get("median_corner_error_mm")
    if median is None or median > 2.0:
        return f"local median disagreement is {median!r} mm (limit 2.00 mm)"
    leave_one_out = diagnostics.get("leave_one_out_validation", {})
    folds = leave_one_out.get("folds", [])
    if leave_one_out.get("status") != "completed" or not folds:
        return "leave-one-out validation did not complete"
    if any(fold.get("status") != "accepted" for fold in folds):
        return "not every leave-one-out fold completed"
    worst = leave_one_out.get("worst_influence", {})
    if worst.get("board_corner_max_mm", np.inf) > 1.0:
        return ("leave-one-out worst board-corner change is "
                f"{worst.get('board_corner_max_mm')!r} mm (limit 1.00 mm)")
    if worst.get("rotation_deg", np.inf) > 0.1:
        return ("leave-one-out worst rotation change is "
                f"{worst.get('rotation_deg')!r} deg (limit 0.100 deg)")
    return None


def _draw_outline_preview(image_path, detection, out_path):
    """Write a quick visual check for the outline candidate used in preflight."""
    image = cv2.imread(str(image_path))
    if image is None:
        return
    candidates = detection.get("candidates", [])
    if candidates:
        corners = np.asarray(candidates[0]["corners_px"], dtype=np.int32)
        cv2.polylines(image, [corners], True, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.putText(image, f"outline candidates: {len(candidates)}",
                    (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
                    cv2.LINE_AA)
    else:
        cv2.putText(image, detection.get("reason", "outline not found"),
                    (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3,
                    cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.resize(image, (960, 540)))


def _detect_outlines(capture_dirs, cameras, intrinsics, preview_dir, manual,
                     cache_path, dataset):
    """Detect outline hypotheses before starting any costly rim refinement."""
    selected_captures = {capture_dir.name for capture_dir in capture_dirs}
    cached_rows = []
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("dataset") == dataset:
                cached_rows = [row for row in cached.get("outline_detections", [])
                               if row.get("candidates")]
                print(f"resuming: {len(cached_rows)} outline detections already recorded",
                      flush=True)
        except (OSError, ValueError):
            cached_rows = []
    rows = [row for row in cached_rows
            if row.get("capture") in selected_captures]
    completed = {(row["capture"], row["camera"]) for row in rows}
    for capture_dir in capture_dirs:
        for camera in cameras:
            if (capture_dir.name, camera) in completed:
                continue
            print(f"{capture_dir.name}/{camera}: detecting full-resolution board outline",
                  flush=True)
            K, D = intrinsics[camera]
            if manual:
                title = f"{capture_dir.name}/{camera}"
                detection = outline_board.select_view(capture_dir / camera, K, D, title)
            else:
                detection = outline_board.detect_view(capture_dir / camera, K, D)
            detection.update(capture=capture_dir.name, camera=camera)
            rows.append(detection)
            cached_rows.append(detection)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"dataset": dataset,
                                               "outline_detections": cached_rows}, indent=2))
            if preview_dir is not None:
                _draw_outline_preview(capture_dir / camera / "color.png", detection,
                                      preview_dir / f"{capture_dir.name}_{camera}.png")
    return rows


def _outline_relative_poses(rows, reference, camera):
    """Return all fixed-rig hypotheses from paired, flip-aware outlines."""
    indexed = {(row["capture"], row["camera"]): row for row in rows}
    candidate_sets, rejected = {}, []
    captures = sorted({row["capture"] for row in rows})
    for capture in captures:
        reference_row = indexed.get((capture, reference))
        camera_row = indexed.get((capture, camera))
        if reference_row is None or camera_row is None:
            rejected.append({"capture": capture, "reason": "missing_outline"})
            continue
        reference_candidates = reference_row.get("candidates", [])
        camera_candidates = camera_row.get("candidates", [])
        if not reference_candidates or not camera_candidates:
            rejected.append({"capture": capture, "reason": "outline_not_found",
                             "reference_reason": reference_row.get("reason"),
                             "camera_reason": camera_row.get("reason")})
            continue
        transforms = []
        for reference_pose in reference_candidates:
            for camera_pose in camera_candidates:
                transforms.append(np.asarray(reference_pose["T_cam_board"]) @
                                  np.linalg.inv(np.asarray(camera_pose["T_cam_board"])))
        candidate_sets[capture] = transforms

    hypotheses = [(capture, transform) for capture, transforms
                  in candidate_sets.items() for transform in transforms]
    if not hypotheses:
        return [], rejected
    best_transform, best_score = None, np.inf
    for _, hypothesis in hypotheses:
        error_per_capture = [min(_corner_error_mm(candidate, hypothesis)
                                 for candidate in transforms)
                             for transforms in candidate_sets.values()]
        score = float(np.median(error_per_capture))
        if score < best_score:
            best_transform, best_score = hypothesis, score
    selected = []
    for capture, transforms in candidate_sets.items():
        selected.append((capture, min(transforms,
                                      key=lambda pose: _corner_error_mm(pose,
                                                                        best_transform))))
    return selected, rejected


def _pose_vector(transform):
    rotation, _ = cv2.Rodrigues(transform[:3, :3])
    return np.r_[rotation.ravel(), transform[:3, 3]]


def _pose_matrix(vector):
    rotation, _ = cv2.Rodrigues(np.asarray(vector[:3], dtype=float))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = vector[3:6]
    return transform


def _selected_outline_observations(rows, reference, camera, T_reference_camera):
    """Select corner-order hypotheses coherent with a provisional rig pose."""
    indexed = {(row["capture"], row["camera"]): row for row in rows}
    observations = []
    for capture in sorted({row["capture"] for row in rows}):
        reference_row = indexed.get((capture, reference))
        camera_row = indexed.get((capture, camera))
        if reference_row is None or camera_row is None:
            continue
        choices = []
        for reference_pose in reference_row.get("candidates", []):
            for camera_pose in camera_row.get("candidates", []):
                T_reference_board = np.asarray(reference_pose["T_cam_board"])
                T_camera_board = np.asarray(camera_pose["T_cam_board"])
                relative = T_reference_board @ np.linalg.inv(T_camera_board)
                choices.append((_corner_error_mm(relative, T_reference_camera),
                                reference_pose, camera_pose))
        if not choices:
            continue
        _, reference_pose, camera_pose = min(choices, key=lambda item: item[0])
        reference_corners = np.roll(np.asarray(reference_pose["corners_px"], float),
                                    -reference_pose["corner_offset"], axis=0)
        camera_corners = np.roll(np.asarray(camera_pose["corners_px"], float),
                                 -camera_pose["corner_offset"], axis=0)
        observations.append({"capture": capture,
                             "T_reference_board": np.asarray(reference_pose["T_cam_board"]),
                             "T_camera_board": np.asarray(camera_pose["T_cam_board"]),
                             "reference_corners_px": reference_corners,
                             "camera_corners_px": camera_corners})
    return observations


def _joint_outline_transform(observations, reference_intrinsics, camera_intrinsics,
                             initial_transform):
    """Fit one rig transform and one board pose per pair to all clicked corners."""
    K_reference, D_reference = reference_intrinsics
    K_camera, D_camera = camera_intrinsics
    board_corners = board_model.outline().astype(np.float64)
    seed = np.r_[_pose_vector(initial_transform),
                 *[_pose_vector(item["T_reference_board"]) for item in observations]]

    def project(transform, K, D):
        rotation = cv2.Rodrigues(transform[:3, :3])[0]
        pixels, _ = cv2.projectPoints(board_corners, rotation, transform[:3, 3], K, D)
        return pixels[:, 0, :]

    def residual(vector):
        T_reference_camera = _pose_matrix(vector[:6])
        T_camera_reference = np.linalg.inv(T_reference_camera)
        blocks = []
        for index, observation in enumerate(observations):
            start = 6 + 6 * index
            T_reference_board = _pose_matrix(vector[start:start + 6])
            T_camera_board = T_camera_reference @ T_reference_board
            blocks.append((project(T_reference_board, K_reference, D_reference) -
                           observation["reference_corners_px"]).ravel())
            blocks.append((project(T_camera_board, K_camera, D_camera) -
                           observation["camera_corners_px"]).ravel())
        return np.concatenate(blocks)

    solution = least_squares(residual, seed, loss="soft_l1", f_scale=2.0,
                             x_scale=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
                             + [0.01, 0.01, 0.01, 0.01, 0.01, 0.01] * len(observations),
                             max_nfev=600)
    transform = _pose_matrix(solution.x[:6])
    board_poses = {
        observation["capture"]: _pose_matrix(solution.x[6 + 6 * index:12 + 6 * index])
        for index, observation in enumerate(observations)
    }
    return transform, float(np.sqrt(np.mean(solution.fun ** 2))), board_poses


def _transform_key(transform, mode="standard"):
    """Stable cache key for local secondary-camera rim refinements."""
    rounded = np.asarray(transform, dtype=float).round(12).tolist()
    payload = {"version": RIM_FIT_VERSION, "mode": mode, "transform": rounded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_rim_fit_cache(cache_path, dataset, transform_key):
    if not cache_path.exists():
        return []
    try:
        cached = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        return []
    if cached.get("version") != RIM_FIT_VERSION or cached.get("dataset") != dataset:
        return []
    rows = cached.get("rows", [])
    if cached.get("secondary_seed_key") == transform_key:
        return rows
    # Reference fits do not depend on the provisional rig, so they remain
    # useful when the outline clicks or their joint solution change.
    return [row for row in rows if row.get("camera") == cached.get("reference_camera")]


def _write_rim_fit_cache(cache_path, dataset, reference_camera, transform_key, rows):
    cache_path.write_text(json.dumps({
        "version": RIM_FIT_VERSION,
        "dataset": dataset,
        "reference_camera": reference_camera,
        "secondary_seed_key": transform_key,
        "rows": rows,
    }, indent=2))


def _per_camera_bias_residual(vector, views, observations, intrinsics, points,
                              normals, level):
    bias = vector[:2]
    blocks = []
    for index, view in enumerate(views):
        pose = vector[2 + 6 * index:8 + 6 * index]
        observation = observations[index]
        indices = observation["idx"]
        if len(indices) == 0:
            continue
        rotation = cv2.Rodrigues(pose[:3])[0]
        model_points = board_model.biased_pts(points[indices], normals[indices],
                                              level[indices], rotation, pose[3:], bias)
        projected, _ = bundle_rims.project_with_normals(
            model_points, normals[indices], pose[:3], pose[3:], intrinsics)
        blocks.append(((projected - observation["q"]) * observation["d"]).sum(1))
    blocks.append(bias / RIM_BIAS_PRIOR_SIGMA_M)
    return np.concatenate(blocks)


def _estimate_camera_rim_bias(capture_dirs, rows, camera, intrinsics, max_mad,
                              minimum_captures):
    """Fit only constant top/floor rim offsets for one camera from raw views."""
    rows_by_capture = {row["capture"]: row for row in rows
                       if row["camera"] == camera and row["mad_px"] <= max_mad}
    points, normals, level, loop = board_model.build_levels()
    views = []
    captures = []
    for capture_dir in capture_dirs:
        row = rows_by_capture.get(capture_dir.name)
        if row is None:
            continue
        view = stationary._joint_view(capture_dir, camera, {camera: intrinsics},
                                      np.asarray(row["T_cam_board"]), points)
        if view is None:
            continue
        views.append(view)
        captures.append(capture_dir.name)
    if len(views) < minimum_captures:
        return None, {"accepted": False,
                      "reason": (f"need at least {minimum_captures} usable {camera} "
                                 f"captures for rim bias; found {len(views)}"),
                      "training_captures": captures}

    poses = [np.r_[view.rvec, view.tvec] for view in views]
    yaws = np.unwrap([pose[2] for pose in poses])
    positions = np.asarray([pose[3:] for pose in poses])
    coverage = {
        "yaw_span_deg": float(np.degrees(yaws.max() - yaws.min())),
        "image_position_span_m": float(np.linalg.norm(
            positions[:, :2].max(axis=0) - positions[:, :2].min(axis=0))),
        "depth_span_mm": float((positions[:, 2].max() - positions[:, 2].min()) * 1000),
        "limitation": ("flat-board captures identify constant rim size offsets from "
                       "yaw, image-position, and camera-depth variation; they do not "
                       "validate tilt-dependent bias terms"),
    }

    intrinsics_vector = stationary._intrinsic_vector(*intrinsics)
    vector = np.r_[np.zeros(2), *[np.r_[view.rvec, view.tvec] for view in views]]
    lower = np.r_[[-RIM_BIAS_BOUND_M, -RIM_BIAS_BOUND_M],
                  np.full(6 * len(views), -np.inf)]
    upper = np.r_[[RIM_BIAS_BOUND_M, RIM_BIAS_BOUND_M],
                  np.full(6 * len(views), np.inf)]
    solution = None
    observations = None
    for search_halfwidth in (6, 4, 4):
        bias = vector[:2]
        for index, view in enumerate(views):
            pose = vector[2 + 6 * index:8 + 6 * index]
            view.rvec, view.tvec = pose[:3], pose[3:]
        observations = bundle_rims.measure(views, intrinsics_vector, points, normals,
                                           search_halfwidth, level, loop, bias)
        if sum(len(item["idx"]) for item in observations) < 200 * len(views):
            return None, {"accepted": False,
                          "reason": "too few raw rim observations for b0 bias fit",
                          "training_captures": captures}
        solution = least_squares(
            _per_camera_bias_residual, vector,
            args=(views, observations, intrinsics_vector, points, normals, level),
            bounds=(lower, upper), loss="soft_l1", f_scale=2.0,
            x_scale=np.r_[[0.0001, 0.0001],
                          np.tile([0.01, 0.01, 0.01, 0.002, 0.002, 0.002], len(views))],
            max_nfev=120)
        if not solution.success or not np.all(np.isfinite(solution.x)):
            return None, {"accepted": False, "reason": "b0 bias optimizer did not converge",
                          "training_captures": captures}
        vector = solution.x

    rank = int(np.linalg.matrix_rank(solution.jac))
    expected_rank = len(vector)
    if rank < expected_rank:
        return None, {"accepted": False,
                      "reason": f"b0 bias fit is rank deficient ({rank}/{expected_rank})",
                      "training_captures": captures}
    bias = vector[:2]
    if np.any(np.abs(bias) >= RIM_BIAS_BOUND_M - 1e-6):
        return None, {"accepted": False, "reason": "b0 bias fit reached its 0.5 mm bound",
                      "training_captures": captures}
    covariance = np.linalg.pinv(solution.jac.T @ solution.jac)
    uncertainty = np.sqrt(np.maximum(np.diag(covariance)[:2], 0.0))
    if np.any(uncertainty > RIM_BIAS_PRIOR_SIGMA_M):
        return None, {"accepted": False, "reason": "b0 bias uncertainty exceeds 0.25 mm",
                      "training_captures": captures,
                      "uncertainty_m": uncertainty.tolist()}
    residual = _per_camera_bias_residual(vector, views, observations, intrinsics_vector,
                                         points, normals, level)
    return bias, {
        "accepted": True,
        "names": ["b0_top", "b0_floor"],
        "bias_m": bias.tolist(),
        "mean_m": float(np.mean(bias)),
        "differential_m": float(bias[1] - bias[0]),
        "training_captures": captures,
        "fit_diagnostics": {"reassociation_schedule_px": [6, 4, 4],
                            "n": int(len(residual) - 2),
                            "rms_px": float(np.sqrt(np.mean(residual[:-2] ** 2))),
                            "mad_px": float(np.median(np.abs(residual[:-2]))),
                            "rank": rank,
                            "uncertainty_m": uncertainty.tolist(),
                            "optimizer_nfev": int(solution.nfev),
                            "capture_coverage": coverage},
    }


def _refit_rows_with_frozen_bias(capture_dirs, rows, camera, bias, intrinsics):
    """Refresh local poses so Stage B starts from bias-aware image evidence."""
    refreshed = []
    by_capture = {row["capture"]: row for row in rows if row["camera"] == camera}
    stationary._set_active_camera(*intrinsics)
    for capture_dir in capture_dirs:
        row = by_capture.get(capture_dir.name)
        if row is None:
            continue
        fitted = fit_rims6.fit_view_from_pose(capture_dir / camera,
                                              np.asarray(row["T_cam_board"]),
                                              constrain=False, frozen_rim_bias=bias)
        if fitted is None:
            continue
        fitted.update(capture=capture_dir.name, camera=camera)
        refreshed.append(fitted)
    return refreshed


def _bundle_rim_residuals(vector, capture_views, observations, intrinsics,
                          residual_scales, points, normals, level, reference_camera,
                          camera_biases):
    """Return scaled edge residuals for one rig and one board pose per capture."""
    T_reference_camera = _pose_matrix(vector[:6])
    T_camera_reference = np.linalg.inv(T_reference_camera)
    blocks = []
    for capture_index, (capture, views) in enumerate(capture_views.items()):
        start = 6 + 6 * capture_index
        T_reference_board = _pose_matrix(vector[start:start + 6])
        for camera, view in views.items():
            T_camera_board = (T_reference_board if camera == reference_camera
                              else T_camera_reference @ T_reference_board)
            pose = _pose_vector(T_camera_board)
            observation = observations[(capture, camera)]
            indices = observation["idx"]
            if len(indices) == 0:
                continue
            rotation = cv2.Rodrigues(pose[:3])[0]
            model_points = board_model.biased_pts(points[indices], normals[indices],
                                                  level[indices], rotation, pose[3:],
                                                  camera_biases[camera])
            projected, _ = bundle_rims.project_with_normals(
                model_points, normals[indices], pose[:3], pose[3:], intrinsics[camera])
            residual = ((projected - observation["q"]) * observation["d"]).sum(1)
            blocks.append(residual / residual_scales[camera])
    return np.concatenate(blocks) if blocks else np.zeros(1)


def _independent_pose_shift_px(reference_row, camera_row, initial_transform,
                               points, normals, level, camera_intrinsics, camera_bias):
    """Measure how far the rig prediction is from the secondary local fit.

    Frozen edge correspondences are valid only when the fixed-rig initializer
    already projects each board rim close to the edge selected by the
    independently fitted secondary image.  This image-space check catches a
    wrong local rim association before it can contaminate the shared solve.
    """
    T_reference_board = np.asarray(reference_row["T_cam_board"], float)
    predicted_camera_board = np.linalg.inv(initial_transform) @ T_reference_board
    local_camera_board = np.asarray(camera_row["T_cam_board"], float)
    predicted_pose = _pose_vector(predicted_camera_board)
    local_pose = _pose_vector(local_camera_board)
    predicted_rotation = cv2.Rodrigues(predicted_pose[:3])[0]
    local_rotation = cv2.Rodrigues(local_pose[:3])[0]
    predicted = bundle_rims.project(board_model.biased_pts(
        points, normals, level, predicted_rotation, predicted_pose[3:], camera_bias),
        predicted_pose[:3], predicted_pose[3:], camera_intrinsics)
    local = bundle_rims.project(board_model.biased_pts(
        points, normals, level, local_rotation, local_pose[3:], camera_bias),
        local_pose[:3], local_pose[3:], camera_intrinsics)
    return float(np.median(np.linalg.norm(predicted - local, axis=1)))


def _bundle_transform_from_synchronized_rims(capture_dirs, rows, reference_camera,
                                             camera, intrinsics, initial_transform,
                                             captures, max_reference_mad,
                                             max_secondary_mad, minimum_pairs,
                                             camera_biases):
    """Jointly refine one fixed rig transform from raw native-resolution rims.

    Local fits provide only initialization.  This solve shares the secondary
    camera transform across all captures while retaining a separate board pose
    for each capture, so every residual is measured in its source raw image.
    """
    indexed = {(row["capture"], row["camera"]): row for row in rows}
    points, normals, level, loop = board_model.build_levels()
    intrinsics_vectors = {
        name: stationary._intrinsic_vector(*intrinsics[name])
        for name in (reference_camera, camera)
    }
    capture_views = {}
    observations = {}
    initial_board_poses = []
    residual_scales = {reference_camera: [], camera: []}
    rejected_captures = []
    for capture_dir in capture_dirs:
        if capture_dir.name not in captures:
            continue
        reference_row = indexed[(capture_dir.name, reference_camera)]
        camera_row = indexed[(capture_dir.name, camera)]
        T_reference_board = np.asarray(reference_row["T_cam_board"], float)
        T_camera_board = np.asarray(camera_row["T_cam_board"], float)
        initial_shift_px = _independent_pose_shift_px(
            reference_row, camera_row, initial_transform, points,
            normals, level, intrinsics_vectors[camera], camera_biases[camera])
        if initial_shift_px > FROZEN_CORRESPONDENCE_MAX_SHIFT_PX:
            rejected_captures.append({
                "capture": capture_dir.name,
                "reason": "independent_local_pose_disagrees_with_rig_initializer",
                "median_projection_shift_px": initial_shift_px,
            })
            continue
        reference_view = stationary._joint_view(
            capture_dir, reference_camera, intrinsics, T_reference_board, points)
        camera_view = stationary._joint_view(
            capture_dir, camera, intrinsics, T_camera_board, points)
        if reference_view is None or camera_view is None:
            rejected_captures.append({
                "capture": capture_dir.name,
                "reason": "could_not_load_raw_image_view",
            })
            continue
        # The special reference key keeps the residual function's transform
        # convention explicit while the output still uses real camera names.
        capture_views[capture_dir.name] = {
            "__reference__": reference_view,
            camera: camera_view,
        }
        initial_board_poses.append(_pose_vector(T_reference_board))
        residual_scales[reference_camera].append(reference_row["mad_px"])
        residual_scales[camera].append(camera_row["mad_px"])

        # Associate the physical rim once at the independent local pose.  The
        # optimizer below moves only the projection; it must not search again,
        # because a wide search can silently switch to a parallel rim.
        for view_camera, view in capture_views[capture_dir.name].items():
            local_pose = (T_reference_board if view_camera == "__reference__"
                          else T_camera_board)
            pose = _pose_vector(local_pose)
            view.rvec, view.tvec = pose[:3], pose[3:]
            observations[(capture_dir.name, view_camera)] = bundle_rims.measure(
                [view], intrinsics_vectors[(reference_camera
                                             if view_camera == "__reference__"
                                             else view_camera)], points, normals,
                4, level, loop, camera_biases[(reference_camera
                                                if view_camera == "__reference__"
                                                else view_camera)])[0]

    if len(capture_views) < minimum_pairs:
        return None, {"accepted": False,
                      "reason": (f"need at least {minimum_pairs} usable "
                                 "frozen-correspondence capture pairs"),
                      "rejected_captures": rejected_captures}

    # Replace the internal reference sentinel with its actual name in all
    # structures passed to the generic projection helpers.
    for views in capture_views.values():
        views[reference_camera] = views.pop("__reference__")
    observations = {
        (capture, reference_camera if view_camera == "__reference__" else view_camera): item
        for (capture, view_camera), item in observations.items()
    }
    scales = {
        name: max(float(np.median(values)), 0.10)
        for name, values in residual_scales.items()
    }
    vector = np.r_[_pose_vector(initial_transform), *initial_board_poses]
    observation_count = sum(len(item["idx"]) for item in observations.values())
    if observation_count < 200 * len(capture_views):
        return None, {"accepted": False,
                      "reason": "too few frozen raw rim observations for bundle adjustment",
                      "observation_count": int(observation_count),
                      "rejected_captures": rejected_captures}
    x_scale = np.tile([0.01, 0.01, 0.01, 0.002, 0.002, 0.002],
                      1 + len(capture_views))
    last_solution = least_squares(
        _bundle_rim_residuals, vector,
        args=(capture_views, observations, intrinsics_vectors, scales, points, normals,
              level, reference_camera, camera_biases),
        loss="soft_l1", f_scale=2.0, x_scale=x_scale, max_nfev=120)
    vector = last_solution.x

    if last_solution is None or not last_solution.success:
        message = "optimizer did not converge"
        if last_solution is not None:
            message = last_solution.message
        return None, {"accepted": False, "reason": message}

    T_reference_camera = _pose_matrix(vector[:6])
    T_camera_reference = np.linalg.inv(T_reference_camera)

    per_camera, per_capture = {}, {}
    all_residuals = []
    for capture_index, (capture, views) in enumerate(capture_views.items()):
        start = 6 + 6 * capture_index
        T_reference_board = _pose_matrix(vector[start:start + 6])
        capture_residuals = []
        for view_camera in views:
            T_camera_board = (T_reference_board if view_camera == reference_camera
                              else T_camera_reference @ T_reference_board)
            pose = _pose_vector(T_camera_board)
            observation = observations[(capture, view_camera)]
            indices = observation["idx"]
            rotation = cv2.Rodrigues(pose[:3])[0]
            model_points = board_model.biased_pts(points[indices], normals[indices],
                                                  level[indices], rotation, pose[3:],
                                                  camera_biases[view_camera])
            projected, _ = bundle_rims.project_with_normals(
                model_points, normals[indices], pose[:3], pose[3:],
                intrinsics_vectors[view_camera])
            residual = ((projected - observation["q"]) * observation["d"]).sum(1)
            all_residuals.append(residual)
            capture_residuals.append(residual)
            per_camera.setdefault(view_camera, []).append(residual)
        merged = np.concatenate(capture_residuals)
        per_capture[capture] = {
            "n": int(len(merged)),
            "rms_px": float(np.sqrt(np.mean(merged ** 2))),
            "mad_px": float(np.median(np.abs(merged))),
            "T_reference_board": T_reference_board.tolist(),
        }
    per_camera_metrics = {}
    for name, residual_sets in per_camera.items():
        merged = np.concatenate(residual_sets)
        per_camera_metrics[name] = {
            "n": int(len(merged)),
            "rms_px": float(np.sqrt(np.mean(merged ** 2))),
            "mad_px": float(np.median(np.abs(merged))),
            "weight_scale_px": scales[name],
        }
    merged = np.concatenate(all_residuals)
    metrics = {
        "accepted": True,
        "method": "fixed_rig_native_4k_bundle",
        "captures": list(capture_views),
        "n": int(len(merged)),
        "rms_px": float(np.sqrt(np.mean(merged ** 2))),
        "mad_px": float(np.median(np.abs(merged))),
        "per_camera": per_camera_metrics,
        "per_capture": per_capture,
        "optimizer_nfev": int(last_solution.nfev),
        "optimizer_cost": float(last_solution.cost),
        "frozen_correspondences": True,
        "correspondence_search_halfwidth_px": 4,
        "maximum_initial_projection_shift_px": FROZEN_CORRESPONDENCE_MAX_SHIFT_PX,
        "rejected_captures": rejected_captures,
    }
    quality_limits = {reference_camera: max_reference_mad,
                      camera: max_secondary_mad}
    failures = [f"{name} MAD {per_camera_metrics[name]['mad_px']:.2f} px "
                f"exceeds {quality_limits[name]:.2f} px"
                for name in quality_limits
                if per_camera_metrics[name]["mad_px"] > quality_limits[name]]
    if failures:
        metrics["accepted"] = False
        metrics["reason"] = "; ".join(failures)
        return None, metrics
    return T_reference_camera, metrics


def _transform_delta(transform, reference):
    """Describe a rig-transform change in directly inspectable units."""
    translation_mm = float(np.linalg.norm(transform[:3, 3] - reference[:3, 3]) * 1000)
    relative_rotation = transform[:3, :3] @ reference[:3, :3].T
    rotation_deg = float(np.degrees(np.linalg.norm(cv2.Rodrigues(relative_rotation)[0])))
    corner_delta = board_corners(transform) - board_corners(reference)
    corner_distances_mm = np.linalg.norm(corner_delta, axis=1) * 1000
    return {
        "translation_mm": translation_mm,
        "rotation_deg": rotation_deg,
        "board_corner_rms_mm": float(np.sqrt(np.mean(corner_distances_mm ** 2))),
        "board_corner_max_mm": float(np.max(corner_distances_mm)),
    }


def _held_out_rim_metrics(capture_dir, rows, reference_camera, camera, intrinsics,
                          T_reference_camera, camera_biases):
    """Fit only a held-out board pose while keeping a candidate rig fixed."""
    indexed = {(row["capture"], row["camera"]): row for row in rows}
    reference_row = indexed[(capture_dir.name, reference_camera)]
    camera_row = indexed[(capture_dir.name, camera)]
    points, normals, level, loop = board_model.build_levels()
    intrinsics_vectors = {
        name: stationary._intrinsic_vector(*intrinsics[name])
        for name in (reference_camera, camera)
    }
    initial_shift_px = _independent_pose_shift_px(
        reference_row, camera_row, T_reference_camera, points,
        normals, level, intrinsics_vectors[camera], camera_biases[camera])
    if initial_shift_px > FROZEN_CORRESPONDENCE_MAX_SHIFT_PX:
        return None, {
            "status": "unavailable",
            "reason": "independent_local_pose_disagrees_with_refit_rig",
            "median_projection_shift_px": initial_shift_px,
        }

    T_reference_board = np.asarray(reference_row["T_cam_board"], float)
    T_camera_board = np.asarray(camera_row["T_cam_board"], float)
    views = {
        reference_camera: stationary._joint_view(
            capture_dir, reference_camera, intrinsics, T_reference_board, points),
        camera: stationary._joint_view(
            capture_dir, camera, intrinsics, T_camera_board, points),
    }
    if any(view is None for view in views.values()):
        return None, {"status": "unavailable", "reason": "could_not_load_raw_image_view"}

    observations = {}
    for name, view in views.items():
        local_pose = T_reference_board if name == reference_camera else T_camera_board
        pose = _pose_vector(local_pose)
        view.rvec, view.tvec = pose[:3], pose[3:]
        observations[name] = bundle_rims.measure(
            [view], intrinsics_vectors[name], points, normals, 4, level, loop,
            camera_biases[name])[0]

    scales = {
        reference_camera: max(float(reference_row["mad_px"]), 0.10),
        camera: max(float(camera_row["mad_px"]), 0.10),
    }
    T_camera_reference = np.linalg.inv(T_reference_camera)

    def residual(vector):
        fitted_reference_board = _pose_matrix(vector)
        blocks = []
        for name in (reference_camera, camera):
            fitted_camera_board = (fitted_reference_board if name == reference_camera
                                   else T_camera_reference @ fitted_reference_board)
            pose = _pose_vector(fitted_camera_board)
            observation = observations[name]
            indices = observation["idx"]
            rotation = cv2.Rodrigues(pose[:3])[0]
            model_points = board_model.biased_pts(points[indices], normals[indices],
                                                  level[indices], rotation, pose[3:],
                                                  camera_biases[name])
            projected, _ = bundle_rims.project_with_normals(
                model_points, normals[indices], pose[:3], pose[3:], intrinsics_vectors[name])
            edge_distance = ((projected - observation["q"]) * observation["d"]).sum(1)
            blocks.append(edge_distance / scales[name])
        return np.concatenate(blocks)

    solution = least_squares(
        residual, _pose_vector(T_reference_board), loss="soft_l1", f_scale=2.0,
        x_scale=[0.01, 0.01, 0.01, 0.002, 0.002, 0.002], max_nfev=80)
    if not solution.success:
        return None, {"status": "unavailable", "reason": str(solution.message)}

    fitted_reference_board = _pose_matrix(solution.x)
    per_camera = {}
    residuals = []
    for name in (reference_camera, camera):
        fitted_camera_board = (fitted_reference_board if name == reference_camera
                               else T_camera_reference @ fitted_reference_board)
        pose = _pose_vector(fitted_camera_board)
        observation = observations[name]
        indices = observation["idx"]
        rotation = cv2.Rodrigues(pose[:3])[0]
        model_points = board_model.biased_pts(points[indices], normals[indices],
                                              level[indices], rotation, pose[3:],
                                              camera_biases[name])
        projected, _ = bundle_rims.project_with_normals(
            model_points, normals[indices], pose[:3], pose[3:], intrinsics_vectors[name])
        edge_distance = ((projected - observation["q"]) * observation["d"]).sum(1)
        residuals.append(edge_distance)
        per_camera[name] = {
            "n": int(len(edge_distance)),
            "rms_px": float(np.sqrt(np.mean(edge_distance ** 2))),
            "mad_px": float(np.median(np.abs(edge_distance))),
        }
    merged = np.concatenate(residuals)
    return fitted_reference_board, {
        "status": "accepted",
        "n": int(len(merged)),
        "rms_px": float(np.sqrt(np.mean(merged ** 2))),
        "mad_px": float(np.median(np.abs(merged))),
        "per_camera": per_camera,
        "optimizer_nfev": int(solution.nfev),
    }


def _leave_one_out_validation(capture_dirs, rows, reference_camera, camera, intrinsics,
                              full_transform, bundle_diagnostics, max_reference_mad,
                              max_secondary_mad, minimum_pairs, camera_biases,
                              training_minimum_pairs=None):
    """Measure whether any one accepted capture materially changes the rig fit.

    This is diagnostic only.  It deliberately does not select or reject the
    full-data transform, because a small calibration set needs human review
    before automated outlier policy can safely alter the rig.
    """
    accepted_captures = list(bundle_diagnostics.get("captures", []))
    if training_minimum_pairs is None:
        training_minimum_pairs = minimum_pairs
    capture_by_name = {capture_dir.name: capture_dir for capture_dir in capture_dirs}
    folds = []
    if len(accepted_captures) < 4:
        reason = "need at least four usable bundle captures for leave-one-out validation"
        for capture in accepted_captures:
            folds.append({"omitted_capture": capture, "status": "unavailable",
                          "reason": reason})
        return {"status": "insufficient_captures", "reason": reason, "folds": folds}

    for omitted_capture in accepted_captures:
        training_captures = set(accepted_captures)
        training_captures.remove(omitted_capture)
        refit_transform, refit_diagnostics = _bundle_transform_from_synchronized_rims(
            capture_dirs, rows, reference_camera, camera, intrinsics, full_transform,
            training_captures, max_reference_mad, max_secondary_mad, training_minimum_pairs,
            camera_biases)
        fold = {"omitted_capture": omitted_capture,
                "training_captures": sorted(training_captures)}
        if refit_transform is None:
            fold.update(status="unavailable", reason=refit_diagnostics["reason"],
                        rig_refit=refit_diagnostics)
            folds.append(fold)
            continue
        fold["rig_delta_from_full"] = _transform_delta(refit_transform, full_transform)
        _, held_out_metrics = _held_out_rim_metrics(
            capture_by_name[omitted_capture], rows, reference_camera, camera,
            intrinsics, refit_transform, camera_biases)
        fold["held_out"] = held_out_metrics
        fold["status"] = ("accepted" if held_out_metrics["status"] == "accepted"
                          else "unavailable")
        folds.append(fold)

    accepted_folds = [fold for fold in folds if fold["status"] == "accepted"]
    if not accepted_folds:
        return {"status": "unavailable", "reason": "no leave-one-out refit completed",
                "folds": folds}
    worst = max(accepted_folds,
                key=lambda fold: fold["rig_delta_from_full"]["board_corner_rms_mm"])
    return {
        "status": "completed",
        "folds": folds,
        "worst_influence": {
            "omitted_capture": worst["omitted_capture"],
            **worst["rig_delta_from_full"],
        },
    }


def _refine_transform_from_synchronized_rims(capture_dirs, reference_camera, camera,
                                              intrinsics, outline_transform, cache_path,
                                              dataset, minimum_pairs,
                                              max_reference_mad, max_secondary_mad,
                                              max_transform_error_mm,
                                              minimum_bias_captures,
                                              frozen_camera_biases=None,
                                              reference_pose_seeds=None,
                                              leave_one_out_minimum_pairs=None):
    """Refine a coarse rig using independently seeded reference-image fits.

    The reference camera is normally globally localized from its own board
    image. Transform-only recalibration instead supplies outline-derived
    seeds, retaining a local image refinement while avoiding a global search.
    The fixed transform is accepted only when the independently image-supported
    poses agree across several board placements.
    """
    seed_key = _transform_key(
        outline_transform,
        "transform_only" if frozen_camera_biases is not None else "standard")
    rows = _load_rim_fit_cache(cache_path, dataset, seed_key)
    completed = {(row["capture"], row["camera"]) for row in rows}
    total = len(capture_dirs) * 2

    def save():
        _write_rim_fit_cache(cache_path, dataset, reference_camera, seed_key, rows)

    for capture_index, capture_dir in enumerate(capture_dirs, start=1):
        reference_row = next((row for row in rows
                              if row["capture"] == capture_dir.name
                              and row["camera"] == reference_camera), None)
        if (reference_row is not None and frozen_camera_biases is not None
                and reference_row["mad_px"] > max_reference_mad):
            rows.remove(reference_row)
            reference_row = None
        if reference_row is None:
            stationary._set_active_camera(*intrinsics[reference_camera])
            print(f"[{2 * capture_index - 1}/{total}] {capture_dir.name}/{reference_camera}: "
                  "independent native 4K rim fit", flush=True)

            def report(stage):
                print(f"[{2 * capture_index - 1}/{total}] {capture_dir.name}/"
                      f"{reference_camera}: {stage}", flush=True)

            pose_seed = (reference_pose_seeds or {}).get(capture_dir.name)
            if pose_seed is None:
                fitted = fit_rims6.fit_view(capture_dir / reference_camera, topk=20,
                                            uv_span=0.55, progress=report,
                                            bright_seed=True,
                                            frozen_rim_bias=(frozen_camera_biases or {}).get(
                                                reference_camera))
            else:
                fitted = fit_rims6.fit_view_from_pose(
                    capture_dir / reference_camera, pose_seed, constrain=False,
                    progress=report,
                    frozen_rim_bias=(frozen_camera_biases or {}).get(reference_camera))
                if fitted is None or fitted["mad_px"] > max_reference_mad:
                    report("outline seed did not pass; falling back to global image search")
                    fitted = fit_rims6.fit_view(
                        capture_dir / reference_camera, topk=20, uv_span=0.55,
                        progress=report, bright_seed=True,
                        frozen_rim_bias=frozen_camera_biases[reference_camera])
            if fitted is not None:
                fitted.update(capture=capture_dir.name, camera=reference_camera)
                rows.append(fitted)
                reference_row = fitted
            save()

        if (capture_dir.name, camera) in completed:
            continue
        if reference_row is None or reference_row["mad_px"] > max_reference_mad:
            print(f"[{2 * capture_index}/{total}] {capture_dir.name}/{camera}: skipped; "
                  "reference image fit did not pass", flush=True)
            continue
        stationary._set_active_camera(*intrinsics[camera])
        T_camera_board = np.linalg.inv(outline_transform) @ np.asarray(
            reference_row["T_cam_board"])
        print(f"[{2 * capture_index}/{total}] {capture_dir.name}/{camera}: "
              "rig-seeded native 4K rim fit", flush=True)
        fitted = fit_rims6.fit_view_from_pose(capture_dir / camera, T_camera_board,
                                              constrain=False,
                                              progress=lambda stage: print(
                                                  f"[{2 * capture_index}/{total}] "
                                                  f"{capture_dir.name}/{camera}: {stage}",
                                                  flush=True),
                                              frozen_rim_bias=(frozen_camera_biases or {}).get(camera))
        if fitted is not None:
            fitted.update(capture=capture_dir.name, camera=camera)
            rows.append(fitted)
        save()

    if frozen_camera_biases is None:
        camera_biases = {}
        bias_diagnostics = {}
        for name, maximum_mad in ((reference_camera, max_reference_mad),
                                  (camera, max_secondary_mad)):
            bias, diagnostics = _estimate_camera_rim_bias(
                capture_dirs, rows, name, intrinsics[name], maximum_mad,
                minimum_bias_captures)
            if bias is None:
                return outline_transform, {"accepted": False,
                                           "reason": f"{name} rim-bias fit rejected: {diagnostics['reason']}",
                                           "rim_biases": bias_diagnostics | {name: diagnostics}}
            camera_biases[name] = bias
            bias_diagnostics[name] = diagnostics
    else:
        camera_biases = {name: np.asarray(frozen_camera_biases[name], float)
                         for name in (reference_camera, camera)}
        bias_diagnostics = {
            name: {"accepted": True, "reused_from_prior": True,
                   "bias_m": camera_biases[name].tolist()}
            for name in camera_biases
        }

    for name in (reference_camera, camera):
        refreshed = _refit_rows_with_frozen_bias(capture_dirs, rows, name,
                                                  camera_biases[name], intrinsics[name])
        rows = [row for row in rows if row["camera"] != name] + refreshed
    save()

    indexed = {(row["capture"], row["camera"]): row for row in rows}
    relative = []
    rejected = []
    for capture_dir in capture_dirs:
        reference_row = indexed.get((capture_dir.name, reference_camera))
        camera_row = indexed.get((capture_dir.name, camera))
        if reference_row is None or camera_row is None:
            rejected.append({"capture": capture_dir.name, "reason": "missing_rim_fit"})
            continue
        if reference_row["mad_px"] > max_reference_mad:
            rejected.append({"capture": capture_dir.name, "reason": "reference_rim_mad",
                             "mad_px": reference_row["mad_px"]})
            continue
        if camera_row["mad_px"] > max_secondary_mad:
            rejected.append({"capture": capture_dir.name, "reason": "secondary_rim_mad",
                             "mad_px": camera_row["mad_px"]})
            continue
        relative.append((capture_dir.name,
                         np.asarray(reference_row["T_cam_board"]) @
                         np.linalg.inv(np.asarray(camera_row["T_cam_board"]))))
    if len(relative) < minimum_pairs:
        return outline_transform, {
            "accepted": False,
            "reason": f"need {minimum_pairs} accepted rim pairs; found {len(relative)}",
            "rejected_captures": rejected,
        }
    state_error, clusters, _ = _multiple_rig_states_error(
        relative, maximum_distance_mm=max_transform_error_mm)
    reference_rows = {
        capture: indexed[(capture, reference_camera)]
        for capture, _ in relative
    }
    diversity = _capture_pose_diversity(relative, reference_rows)
    if diversity["warnings"]:
        print(f"{reference_camera} <- {camera}: capture pose diversity warning: "
              f"{'; '.join(diversity['warnings'])} "
              f"(normal span {diversity['normal_span_deg']:.1f} deg, "
              f"depth span {diversity['depth_span_mm']:.1f} mm)", flush=True)
    if state_error is not None:
        return outline_transform, {
            "accepted": False,
            "reason": state_error,
            "rejected_captures": rejected,
            "local_transform_clusters": clusters,
            "capture_pose_diversity": diversity,
        }
    try:
        transform, _, errors, threshold, kept = estimate_transform(relative, minimum_pairs)
    except RuntimeError as exc:
        return outline_transform, {
            "accepted": False,
            "reason": str(exc),
            "rejected_captures": rejected,
        }
    median_error = float(np.median(errors))
    local_diagnostics = {
        "accepted_captures": [capture for capture, _ in kept],
        "rejected_captures": rejected,
        "local_transform_clusters": clusters,
        "capture_pose_diversity": diversity,
        "pair_corner_errors_mm": {capture: float(error)
                                  for (capture, _), error in zip(relative, errors)},
        "outlier_threshold_mm": threshold,
        "median_corner_error_mm": median_error,
        "within_previous_limit": median_error <= max_transform_error_mm,
    }
    bundle_transform, bundle_diagnostics = _bundle_transform_from_synchronized_rims(
        capture_dirs, rows, reference_camera, camera, intrinsics, transform,
        {capture for capture, _ in kept}, max_reference_mad, max_secondary_mad,
        minimum_pairs, camera_biases)
    if bundle_transform is None:
        return outline_transform, {
            "accepted": False,
            "reason": f"native 4K rig bundle rejected ({bundle_diagnostics['reason']})",
            "local_transform_diagnostics": local_diagnostics,
            "bundle": bundle_diagnostics,
        }
    leave_one_out = _leave_one_out_validation(
        capture_dirs, rows, reference_camera, camera, intrinsics, bundle_transform,
        bundle_diagnostics, max_reference_mad, max_secondary_mad, minimum_pairs,
        camera_biases, leave_one_out_minimum_pairs)
    bundle_diagnostics["leave_one_out_validation"] = leave_one_out
    if leave_one_out["status"] == "completed":
        worst = leave_one_out["worst_influence"]
        print(f"{reference_camera} <- {camera}: leave-one-out worst influence "
              f"{worst['omitted_capture']} changes rig by "
              f"{worst['board_corner_rms_mm']:.2f} mm RMS / "
              f"{worst['board_corner_max_mm']:.2f} mm max at board corners "
              f"({worst['rotation_deg']:.3f} deg, "
              f"{worst['translation_mm']:.2f} mm translation)", flush=True)
    else:
        print(f"{reference_camera} <- {camera}: leave-one-out validation unavailable "
              f"({leave_one_out['reason']})", flush=True)
    bundle_diagnostics["local_transform_diagnostics"] = local_diagnostics
    bundle_diagnostics["rim_biases"] = bias_diagnostics
    return bundle_transform, bundle_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "run_calibration")
    parser.add_argument("--reference-camera", default="mega")
    parser.add_argument("--out", type=Path,
                        default=HERE / "out" / "camera_rig_calib.json")
    parser.add_argument("--preview-dir", type=Path,
                        help="write board-outline preflight previews here")
    parser.add_argument("--manual-outlines", action="store_true",
                        help="click four board corners in each calibration image")
    parser.add_argument("--restart-outlines", action="store_true",
                        help="discard recorded outline detections and start again")
    parser.add_argument("--max-rig-error-mm", type=float, default=20.0,
                        help="reject a calibration when its cross-pose error exceeds this")
    parser.add_argument("--minimum-pairs", type=int, default=5)
    parser.add_argument("--minimum-bias-captures", type=int, default=8,
                        help="minimum usable raw views per camera for the b0 rim-bias fit")
    parser.add_argument("--recalibrate-transform-from", type=Path,
                        help=("accepted current-schema rig calibration whose intrinsics and "
                              "per-camera rim biases are reused; estimates only a new rig transform"))
    parser.add_argument("--max-camera-color-span-ms", type=float, default=75.0,
                        help=("report spans above this diagnostic value; do not reject "
                              "them because the board is stationary"))
    parser.add_argument("--capture", action="append", default=[], metavar="CAPTURE_NAME",
                        help="use only this synchronized capture (repeatable)")
    parser.add_argument("--skip-rim-refinement", action="store_true",
                        help="save the lower-precision outline-only rig calibration")
    parser.add_argument("--max-rim-mad-px", type=float, default=1.0,
                        help="maximum MAD for a secondary-camera local rim fit")
    parser.add_argument("--max-reference-rim-mad-px", type=float, default=0.7,
                        help="maximum MAD for an independently fitted reference image")
    parser.add_argument("--max-rim-rig-error-mm", type=float, default=4.0,
                        help="report when independent local transforms disagree by this much")
    parser.add_argument("--rim-cache", type=Path,
                        help="reuse or write independent rim fits at this path")
    args = parser.parse_args()
    if args.minimum_pairs < 3:
        raise SystemExit("--minimum-pairs must be at least 3")
    if (args.recalibrate_transform_from is not None and
            args.minimum_pairs < 5):
        raise SystemExit("transform-only recalibration requires --minimum-pairs of at least 5")
    if (args.recalibrate_transform_from is not None and
            args.out.resolve() == args.recalibrate_transform_from.resolve()):
        raise SystemExit("--out must not overwrite --recalibrate-transform-from")
    if args.minimum_bias_captures < 5:
        raise SystemExit("--minimum-bias-captures must be at least 5")
    if args.skip_rim_refinement:
        raise SystemExit("--skip-rim-refinement cannot produce the required rim-bias model")

    capture_dirs = stationary._capture_dirs(args.data)
    available_captures = {capture_dir.name for capture_dir in capture_dirs}
    if len(args.capture) != len(set(args.capture)):
        raise SystemExit("--capture names must be unique")
    missing_captures = sorted(set(args.capture) - available_captures)
    if missing_captures:
        raise SystemExit(f"requested capture(s) not found: {', '.join(missing_captures)}")
    if args.capture:
        requested_captures = set(args.capture)
        capture_dirs = [capture_dir for capture_dir in capture_dirs
                        if capture_dir.name in requested_captures]
    cameras = stationary._camera_names(capture_dirs)
    if not capture_dirs:
        raise SystemExit(f"no capture_* directories in {args.data}")
    if len(capture_dirs) < args.minimum_pairs:
        raise SystemExit(f"need at least {args.minimum_pairs} selected captures; found "
                         f"{len(capture_dirs)}")
    try:
        maximum_observed_span_ms = stationary._validate_capture_timing(
            capture_dirs, args.max_camera_color_span_ms)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.reference_camera not in cameras:
        raise SystemExit(f"reference camera {args.reference_camera!r} not in {cameras}")
    if len(cameras) < 2:
        raise SystemExit("rig calibration needs at least two synchronized cameras")
    if maximum_observed_span_ms > args.max_camera_color_span_ms:
        print(f"notice: maximum inter-camera colour span is "
              f"{maximum_observed_span_ms:.1f} ms, above the diagnostic "
              f"value {args.max_camera_color_span_ms:.1f} ms; "
              "the stationary-board calibration accepts this", flush=True)

    intrinsics = {camera: stationary._camera_intrinsics(
        capture_dirs[0] / camera / "camera_info.json") for camera in cameras}
    camera_info_paths = {camera: capture_dirs[0] / camera / "camera_info.json"
                         for camera in cameras}
    prior_calibration = prior_poses = prior_biases = prior_provenance = None
    if args.recalibrate_transform_from is not None:
        try:
            prior_calibration, prior_poses, prior_biases, prior_provenance = (
                _load_transform_recalibration_prior(
                    args.recalibrate_transform_from, args.reference_camera, cameras,
                    intrinsics, camera_info_paths))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot use transform recalibration prior: {exc}") from exc
        print(f"transform-only recalibration from {prior_provenance['path']}; "
              "reusing validated per-camera rim biases", flush=True)
    preview_dir = args.preview_dir
    if preview_dir is None:
        preview_dir = args.out.parent / f"{args.out.stem}_outline_previews"
    cache_path = args.out.with_suffix(".outline_detections.json")
    if args.restart_outlines and cache_path.exists():
        cache_path.unlink()
    print(f"{len(capture_dirs)} synchronized calibration captures; "
          "checking board outlines before rim fitting", flush=True)
    outline_rows = _detect_outlines(capture_dirs, cameras, intrinsics, preview_dir,
                                    args.manual_outlines, cache_path,
                                    str(args.data.resolve()))

    transforms = {args.reference_camera: np.eye(4).tolist()}
    diagnostics = {}
    rim_bias_entries = {}
    for camera in cameras:
        if camera == args.reference_camera:
            continue
        relative, rejected = _outline_relative_poses(outline_rows,
                                                      args.reference_camera, camera)
        try:
            outline_transform, initial_errors, final_errors, threshold, kept = estimate_transform(
                relative, args.minimum_pairs)
        except RuntimeError as exc:
            raise SystemExit(f"{args.reference_camera} <- {camera}: {exc}") from exc
        outline_median_error = float(np.median(final_errors))
        if outline_median_error > args.max_rig_error_mm:
            raise SystemExit(
                f"{args.reference_camera} <- {camera}: outline hypotheses still disagree "
                f"by {outline_median_error:.2f} mm (limit {args.max_rig_error_mm:.2f} mm)")
        observations = _selected_outline_observations(outline_rows,
                                                       args.reference_camera,
                                                       camera, outline_transform)
        if len(observations) < args.minimum_pairs:
            raise SystemExit(f"{args.reference_camera} <- {camera}: only "
                             f"{len(observations)} usable outline pairs")
        transform, joint_rms_px, outline_board_poses = _joint_outline_transform(
            observations, intrinsics[args.reference_camera], intrinsics[camera],
            outline_transform)
        outline_joint_transform = transform.copy()
        rim_diagnostics = {"accepted": False, "reason": "disabled"}
        if not args.skip_rim_refinement:
            print(f"{args.reference_camera} <- {camera}: refining from independent "
                  "reference fits and rig-seeded secondary fits", flush=True)
            rim_cache = args.rim_cache or args.out.with_suffix(".rim_fits.json")
            transform, rim_diagnostics = _refine_transform_from_synchronized_rims(
                capture_dirs, args.reference_camera, camera, intrinsics,
                outline_joint_transform, rim_cache, str(args.data.resolve()),
                args.minimum_pairs, args.max_reference_rim_mad_px,
                args.max_rim_mad_px, args.max_rim_rig_error_mm,
                args.minimum_bias_captures,
                frozen_camera_biases=(prior_biases if prior_biases is not None else None),
                reference_pose_seeds=(outline_board_poses if prior_biases is not None else None),
                leave_one_out_minimum_pairs=(4 if prior_biases is not None else None))
            if not rim_diagnostics["accepted"]:
                raise SystemExit(
                    f"{args.reference_camera} <- {camera}: rim refinement rejected "
                    f"({rim_diagnostics['reason']}); no calibration was written")
            if prior_biases is not None:
                failure = _fast_recalibration_gates(rim_diagnostics)
                if failure is not None:
                    raise SystemExit(
                        f"{args.reference_camera} <- {camera}: transform-only recalibration "
                        f"rejected ({failure}); no calibration was written")
                prior_entry = prior_calibration["rim_bias_model"]["per_camera"]
                rim_diagnostics["rim_biases"] = {
                    name: prior_entry[name] for name in (args.reference_camera, camera)}
            rim_bias_entries.update(rim_diagnostics["rim_biases"])
        median_error = outline_median_error
        transforms[camera] = transform.tolist()
        kept_names = {capture for capture, _ in kept}
        for (capture, _), error in zip(relative, initial_errors):
            if capture not in kept_names:
                rejected.append({"capture": capture, "reason": "rig_transform_outlier",
                                 "medoid_corner_error_mm": float(error)})
        diagnostics[camera] = {
            "accepted_captures": [capture for capture, _ in kept],
            "rejected_captures": rejected,
            "pair_corner_errors_mm": {capture: float(error)
                                      for (capture, _), error in zip(relative, final_errors)},
            "medoid_outlier_threshold_mm": threshold,
            "median_corner_error_mm": median_error,
            "max_corner_error_mm": float(np.max(final_errors)),
            "outline_median_corner_error_mm": outline_median_error,
            "outline_outlier_threshold_mm": threshold,
            "joint_corner_rms_px": joint_rms_px,
            "rim_refinement": rim_diagnostics,
            "leave_one_out_validation": rim_diagnostics.get(
                "leave_one_out_validation",
                {"status": "unavailable", "reason": "rim refinement was not accepted"}),
        }
        if prior_poses is not None:
            diagnostics[camera]["transform_delta_from_prior"] = _transform_delta(
                transform, prior_poses[camera])
        print(f"{args.reference_camera} <- {camera}: {len(kept)}/{len(relative)} "
              f"outline pairs used; joint corner RMS {joint_rms_px:.2f} px", flush=True)

    output = {
        "version": CALIBRATION_VERSION,
        "calibration_status": "accepted",
        "calibration_dataset": str(args.data.resolve()),
        "selected_captures": [capture_dir.name for capture_dir in capture_dirs],
        "frame": f"{args.reference_camera}_camera",
        "reference_camera": args.reference_camera,
        "camera_intrinsics": _intrinsics_provenance(capture_dirs[0], cameras, intrinsics),
        "rim_bias_model": {"version": 1, "per_camera": rim_bias_entries,
                           "description": "independent per-camera constant top/floor rim offsets"},
        "camera_poses_in_reference": transforms,
        "camera_diagnostics": diagnostics,
        "outline_previews": str(preview_dir.resolve()),
        "outline_detections": outline_rows,
        "fit_quality_gate": {"max_rig_error_mm": args.max_rig_error_mm,
                              "minimum_pairs": args.minimum_pairs},
        "capture_timing": {
            "policy": "diagnostic_only_stationary_board",
            "diagnostic_camera_color_span_threshold_ms": args.max_camera_color_span_ms,
            "maximum_observed_camera_color_span_ms": maximum_observed_span_ms,
        },
        "created_unix_s": time.time(),
    }
    if prior_provenance is not None:
        output["transform_recalibration"] = {
            "mode": "transform_only",
            "prior": prior_provenance,
            "biases_reused": True,
            "validation": {
                "local_median_disagreement_mm_max": 2.0,
                "leave_one_out_required": "all folds completed",
                "leave_one_out_corner_max_mm": 1.0,
                "leave_one_out_rotation_deg": 0.1,
            },
            "old_to_new_transform_delta": {
                camera: diagnostics[camera]["transform_delta_from_prior"]
                for camera in diagnostics
                if "transform_delta_from_prior" in diagnostics[camera]
            },
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(f"wrote rig calibration to {args.out}", flush=True)


if __name__ == "__main__":
    main()
