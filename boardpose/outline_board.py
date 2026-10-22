#!/usr/bin/env python3
"""Detect the outer rectangle of an ATB board and turn it into pose seeds.

The transparent board has two nearby outline edges, so this module deliberately
uses the outline only as an initializer.  The rim fitter remains responsible
for sub-pixel metrology.  Unlike the rim pattern, a rectangle is invariant to
a 180 degree turn.  ``detect_view`` consequently returns both valid board
frame conventions for every observed rectangle; rig calibration must choose
the convention using consistency across synchronized captures.
"""

from pathlib import Path

import cv2
import numpy as np

import board_model


# The detector is intentionally permissive: a false rectangle is cheaper than
# missing the board, because calibration scores candidates against the fixed
# camera transform.  These gates merely remove shapes that cannot plausibly be
# the complete 320 x 250 mm plate.
MIN_AREA_FRACTION = 0.003
MAX_AREA_FRACTION = 0.90
MIN_SIDE_PIXELS = 80.0
MAX_CANDIDATES = 8
EDGE_DISTANCE_CLIP_PX = 12.0


def _order_corners(corners):
    """Return clockwise image corners starting at the top-left corner."""
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    centre = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0])
    ordered = corners[np.argsort(angles)]
    # Image coordinates have y down, so this order is clockwise.  Give PnP a
    # stable starting corner; the alternative board convention is added later.
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def _polygon_aspect_ratio(corners):
    lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    if lengths.min() < 1e-9:
        return np.inf
    return float((lengths[0] + lengths[2]) / (lengths[1] + lengths[3]))


def _edge_distance(image):
    """Return a full-resolution edge map and its distance transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    edges = cv2.Canny(gray, 25, 90, L2gradient=True)
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    return edges, distance


def _edge_score(distance, corners, samples_per_side=180):
    """Fractional support for the four full-resolution rectangle sides."""
    height, width = distance.shape
    samples = []
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        count = max(24, int(samples_per_side))
        alpha = np.linspace(0.0, 1.0, count)
        samples.append(start[None, :] * (1.0 - alpha[:, None]) +
                       end[None, :] * alpha[:, None])
    samples = np.concatenate(samples)
    x = np.clip(np.rint(samples[:, 0]).astype(np.int32), 0, width - 1)
    y = np.clip(np.rint(samples[:, 1]).astype(np.int32), 0, height - 1)
    # 1 means each side is on an edge, 0 means all samples are at least the
    # clip distance away.  This remains useful with the acrylic double edge.
    return float(1.0 - np.minimum(distance[y, x], EDGE_DISTANCE_CLIP_PX).mean()
                 / EDGE_DISTANCE_CLIP_PX)


def _contour_quadrilaterals(edges):
    """Return unique, convex quadrilaterals proposed by Canny contours."""
    height, width = edges.shape
    image_area = float(height * width)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    proposals = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 4 * MIN_SIDE_PIXELS:
            continue
        # The transparent sheet often creates two almost-identical contours;
        # two approximation tolerances retain both until edge scoring chooses.
        for epsilon_fraction in (0.008, 0.012, 0.018, 0.026):
            polygon = cv2.approxPolyDP(contour, epsilon_fraction * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            corners = _order_corners(polygon[:, 0, :])
            area = abs(cv2.contourArea(corners.astype(np.float32)))
            if not MIN_AREA_FRACTION * image_area <= area <= MAX_AREA_FRACTION * image_area:
                continue
            side_lengths = np.linalg.norm(
                np.roll(corners, -1, axis=0) - corners, axis=1)
            if side_lengths.min() < MIN_SIDE_PIXELS:
                continue
            aspect = _polygon_aspect_ratio(corners)
            # Perspective changes the image aspect ratio, but not by an order
            # of magnitude for a visible planar rectangle.
            if not 0.35 <= aspect <= 2.9:
                continue
            proposals.append(corners)
    return _deduplicate_quadrilaterals(proposals)


def _line_parameters(segment):
    """Return canonical normal-form line parameters and segment length."""
    x0, y0, x1, y1 = np.asarray(segment, dtype=float)
    direction = np.array([x1 - x0, y1 - y0])
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return None
    normal = np.array([-direction[1], direction[0]]) / length
    if normal[0] < 0 or (abs(normal[0]) < 1e-9 and normal[1] < 0):
        normal = -normal
    return normal, float(normal @ [x0, y0]), length


def _representative_lines(edges):
    """Extract and de-duplicate long line segments from the full 4K edge map."""
    raw = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=90,
                           minLineLength=MIN_SIDE_PIXELS,
                           maxLineGap=28)
    if raw is None:
        return []
    lines = []
    cos_two_degrees = float(np.cos(np.deg2rad(2.0)))
    for segment in raw[:, 0, :]:
        parsed = _line_parameters(segment)
        if parsed is None:
            continue
        normal, rho, length = parsed
        # Hough returns many fragments of one physical edge.  Keep its longest
        # representative, but preserve the acrylic's nearby top and bottom
        # outlines as separate candidates.
        replacement = None
        for index, (old_normal, old_rho, old_length) in enumerate(lines):
            if abs(float(normal @ old_normal)) > cos_two_degrees and abs(rho - old_rho) < 8:
                replacement = index
                break
        if replacement is None:
            lines.append((normal, rho, length))
        elif length > lines[replacement][2]:
            lines[replacement] = (normal, rho, length)
    lines.sort(key=lambda item: item[2], reverse=True)
    # The pair-of-pairs search grows quickly. Long lines cover the board
    # perimeter well enough; retaining more lines mostly creates rectangles
    # from unrelated table texture.
    return lines[:24]


def _intersection(first, second):
    normals = np.stack([first[0], second[0]])
    if abs(np.linalg.det(normals)) < 0.08:
        return None
    return np.linalg.solve(normals, np.array([first[1], second[1]]))


def _hough_quadrilaterals(edges):
    """Propose rectangles from two pairs of approximately parallel lines.

    Contours frequently do not close around transparent acrylic.  This fallback
    does not require corners to be connected in Canny; line support is scored
    afterwards against the original full-resolution edge map.
    """
    lines = _representative_lines(edges)
    if len(lines) < 4:
        return []
    height, width = edges.shape
    image_area = float(height * width)
    parallel_pairs = []
    for first_index, first in enumerate(lines):
        for second in lines[first_index + 1:]:
            if abs(float(first[0] @ second[0])) < np.cos(np.deg2rad(9.0)):
                continue
            separation = abs(first[1] - second[1])
            if separation < MIN_SIDE_PIXELS or separation > max(width, height):
                continue
            parallel_pairs.append((first, second))

    proposals = []
    for first_pair_index, first_pair in enumerate(parallel_pairs):
        first_normal = first_pair[0][0]
        for second_pair in parallel_pairs[first_pair_index + 1:]:
            # Families must be meaningfully different.  A board viewed at a
            # shallow angle can have far-from-orthogonal image sides, so do not
            # impose a right-angle image constraint.
            if abs(float(first_normal @ second_pair[0][0])) > np.cos(np.deg2rad(25.0)):
                continue
            corners = [_intersection(first_pair[0], second_pair[0]),
                       _intersection(first_pair[1], second_pair[0]),
                       _intersection(first_pair[1], second_pair[1]),
                       _intersection(first_pair[0], second_pair[1])]
            if any(corner is None for corner in corners):
                continue
            corners = _order_corners(np.asarray(corners))
            if np.any(corners[:, 0] < -0.15 * width) or np.any(corners[:, 0] > 1.15 * width):
                continue
            if np.any(corners[:, 1] < -0.15 * height) or np.any(corners[:, 1] > 1.15 * height):
                continue
            area = abs(cv2.contourArea(corners.astype(np.float32)))
            if not MIN_AREA_FRACTION * image_area <= area <= MAX_AREA_FRACTION * image_area:
                continue
            side_lengths = np.linalg.norm(
                np.roll(corners, -1, axis=0) - corners, axis=1)
            if side_lengths.min() < MIN_SIDE_PIXELS:
                continue
            aspect = _polygon_aspect_ratio(corners)
            if 0.30 <= aspect <= 3.2:
                proposals.append(corners)
    return _deduplicate_quadrilaterals(proposals)


def _deduplicate_quadrilaterals(proposals, corner_distance_px=16.0):
    """Collapse the top/bottom acrylic contours and repeated approximations."""
    unique = []
    for corners in proposals:
        if any(np.sqrt(((corners - old) ** 2).sum(axis=1).mean()) < corner_distance_px
               for old in unique):
            continue
        unique.append(corners)
    return unique


def _solve_pose(corners, K, D):
    """Solve a positive-depth pose for one ordered outline correspondence."""
    object_points = board_model.outline().astype(np.float64)
    ok, rotation, translation = cv2.solvePnP(
        object_points, corners.astype(np.float64), np.asarray(K, dtype=np.float64),
        np.asarray(D, dtype=np.float64).reshape(-1, 1), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or translation[2, 0] <= 0:
        return None
    projected, _ = cv2.projectPoints(object_points, rotation, translation,
                                     np.asarray(K, dtype=np.float64),
                                     np.asarray(D, dtype=np.float64).reshape(-1, 1))
    reprojection_error = float(np.sqrt(
        ((projected[:, 0, :] - corners) ** 2).sum(axis=1).mean()))
    rotation_matrix, _ = cv2.Rodrigues(rotation)
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation[:, 0]
    return transform, reprojection_error


def _pose_candidates(quadrilaterals, K, D, distance, max_candidates):
    candidates = []
    for corners in quadrilaterals:
        edge_score = _edge_score(distance, corners)
        # The outline contains no marked corner.  All four cyclic mappings are
        # geometrically valid; offsets 0/2 and 1/3 are the two possible
        # long-side assignments, with each pair differing by a 180 degree
        # board-frame flip.  Do not assume image top-left is CAD min/min.
        for corner_offset in range(4):
            mapped_corners = np.roll(corners, -corner_offset, axis=0)
            solved = _solve_pose(mapped_corners, K, D)
            if solved is None:
                continue
            transform, reprojection_error = solved
            confidence = edge_score * min(1.0, 3.0 / max(reprojection_error, 1e-6))
            candidates.append({
                "T_cam_board": transform.tolist(),
                "flip_180": corner_offset >= 2,
                "corner_offset": corner_offset,
                "corners_px": corners.tolist(),
                "edge_score": edge_score,
                "confidence": confidence,
                "reprojection_error_px": reprojection_error,
            })
    candidates.sort(key=lambda candidate: candidate["confidence"], reverse=True)
    return candidates[:max(1, int(max_candidates))]


def detect_image(image, K, D, max_candidates=MAX_CANDIDATES):
    """Find board-outline pose hypotheses in one BGR image.

    Returns a mapping with ``candidates`` and ``reason``.  Each candidate has
    ``T_cam_board`` (a JSON-friendly 4x4 matrix), ``flip_180``, ``corners_px``,
    ``edge_score``, ``confidence`` and ``reprojection_error_px``.  Candidates
    occur in pairs: the first uses the detected rectangle convention and the
    second has an explicit 180 degree board-frame flip.
    """
    if image is None or image.ndim != 3:
        return {"candidates": [], "reason": "could not load a colour image"}
    if image.shape[0] < 2 or image.shape[1] < 2:
        return {"candidates": [], "reason": "image is too small"}

    edges, distance = _edge_distance(image)
    quadrilaterals = _contour_quadrilaterals(edges)
    if not quadrilaterals:
        quadrilaterals = _hough_quadrilaterals(edges)
    if not quadrilaterals:
        return {"candidates": [], "reason": "no plausible rectangular outline found"}

    candidates = _pose_candidates(quadrilaterals, K, D, distance, max_candidates)

    candidates.sort(key=lambda candidate: candidate["confidence"], reverse=True)
    candidates = candidates[:max(1, int(max_candidates))]
    if not candidates:
        return {"candidates": [], "reason": "rectangle pose solve failed"}
    return {"candidates": candidates, "reason": None,
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "edge_pixels": int(np.count_nonzero(edges))}


def detect_view(view_dir, K, D, max_candidates=MAX_CANDIDATES):
    """Detect full-resolution board-outline pose hypotheses in ``view_dir``."""
    view_dir = Path(view_dir)
    image = cv2.imread(str(view_dir / "color.png"), cv2.IMREAD_COLOR)
    return detect_image(image, K, D, max_candidates=max_candidates)


def select_view(view_dir, K, D, window_name):
    """Ask the operator to click the four outer board corners in one image.

    Click corners in any order. Press Enter to accept four clicks, ``r`` to
    retry, or ``q``/Escape to abort calibration. The solve retains all cyclic
    corner mappings, so this does not ask the operator to determine board yaw.
    """
    view_dir = Path(view_dir)
    image = cv2.imread(str(view_dir / "color.png"), cv2.IMREAD_COLOR)
    if image is None:
        return {"candidates": [], "reason": "could not load a colour image"}
    height, width = image.shape[:2]
    scale = min(1.0, 1400.0 / width, 900.0 / height)
    points = []

    def click(event, x, y, _flags, _data):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append(np.array([x / scale, y / scale]))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click)
    try:
        while True:
            display = cv2.resize(image, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)
            for index, point in enumerate(points, start=1):
                pixel = tuple(np.rint(point * scale).astype(int))
                cv2.circle(display, pixel, 8, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(display, str(index), pixel, cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "Click 4 board corners; Enter=accept r=reset q=abort",
                        (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xff
            if key in (27, ord("q")):
                return {"candidates": [], "reason": "operator aborted"}
            if key == ord("r"):
                points.clear()
            if key in (10, 13) and len(points) == 4:
                break
    finally:
        cv2.destroyWindow(window_name)

    corners = _order_corners(np.asarray(points))
    _, distance = _edge_distance(image)
    candidates = _pose_candidates([corners], K, D, distance, 4)
    if not candidates:
        return {"candidates": [], "reason": "manual corner pose solve failed"}
    return {"candidates": candidates, "reason": None,
            "image_size": [width, height], "method": "manual_corners"}
