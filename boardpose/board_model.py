#!/usr/bin/env python3
"""
board_model.py - metrology features for the NIST ATB board, taken from the STL.

The STL is in centimetres: 320 x 250 x 25 mm, holes chamfered (top face at
z = 25 mm, chamfer shoulder at 23 mm). The features used for pose are the
20 hole rims on the TOP FACE, which lie at exactly z = 25 mm -- the same plane
CORNER_3D already assumes.

Deliberately excluded:
  * the outer rectangle. The plate is 25 mm of transparent acrylic, so its
    outline is a double edge (top face and bottom face) separated by up to
    16 px at 4K. That is what wrecked the earlier edge-search metric.
  * the engraved "ManipulationNet" text: sub-millimetre relief, low contrast.

20 rims x ~150 boundary points spread over the full 320 x 250 mm gives roughly
3000 well-distributed constraints, against 4 extrapolated corners today.
"""
from pathlib import Path

import numpy as np
import trimesh

STL = Path(__file__).resolve().parent.parent / "nist_atb_m1.stl"
Z_TOP = 0.025          # m, board top face (matches CORNER_3D)
SPACING = 0.0010       # m, arc-length resampling of each rim

RIM_BIAS_NAMES = ["b0_top", "b0_floor", "b1_top", "b1_floor",
                  "b2_top", "b2_floor", "b3_top", "b3_floor"]
NBIAS = len(RIM_BIAS_NAMES)

# Calibrated by bundle_rims.py; see rim_bias(). Metres of offset along each
# rim's outward in-plane normal. b1 is held at zero on purpose -- it is
# fittable and must not be fitted, see WHY b1 IS ZERO.
RIM_BIAS0 = np.array([1.604e-4, -1.124e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

_MESH = None
_CACHE = {}


def _signed_area(polygon):
    """Shoelace area; the SIGN carries the winding direction."""
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _resample(polygon, spacing):
    """Uniform arc-length resampling of a closed polygon."""
    closed = np.vstack([polygon, polygon[:1]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    point_count = max(24, int(round(arc_length[-1] / spacing)))
    wanted = np.linspace(0, arc_length[-1], point_count, endpoint=False)
    return np.stack([np.interp(wanted, arc_length, closed[:, 0]),
                     np.interp(wanted, arc_length, closed[:, 1])], 1)


def _outward_normals(polygon, signed_area):
    """Unit in-plane normals pointing out of the material, i.e. into the hole."""
    tangents = np.roll(polygon, -1, 0) - np.roll(polygon, 1, 0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], 1)
    is_counter_clockwise = signed_area > 0
    return -normals if is_counter_clockwise else normals


def _mesh():
    """The STL, parsed once per process.

    Cached because build_levels() calls build() twice: 753 redundant parses at
    128 ms over the 251-view run, ~3% of it.
    """
    global _MESH
    if _MESH is None:
        _MESH = trimesh.load(STL)
    return _MESH


def build(z_level=2.5, min_area=0.8, max_area=20.0, max_absy=10.0):
    """Return (pts Nx2 in m, outward normals Nx2, loop id N) for the hole rims.

    Results are memoised per parameter set. Copies are returned so a caller
    that mutates them cannot corrupt the cache.
    """
    key = (z_level, min_area, max_area, max_absy)
    if key in _CACHE:
        return tuple(a.copy() for a in _CACHE[key])
    rim_points, rim_normals, rim_ids = [], [], []
    for contour in _contours_at_height(z_level):
        signed_area = _signed_area(contour)
        if not _is_hole_rim(contour, signed_area, min_area, max_area, max_absy):
            continue
        resampled = _resample(contour, SPACING * 100.0) / 100.0   # cm -> m
        rim_points.append(resampled)
        rim_normals.append(_outward_normals(resampled, signed_area))
        rim_ids.append(np.full(len(resampled), len(rim_points) - 1))

    if not rim_points:
        raise RuntimeError("no rims found")
    rims = (np.concatenate(rim_points), np.concatenate(rim_normals),
            np.concatenate(rim_ids))
    _CACHE[key] = rims
    return tuple(a.copy() for a in rims)


def _contours_at_height(z_level):
    """Closed xy polygons bounding the faces that lie flat at `z_level`."""
    mesh = _mesh()
    face_z = mesh.vertices[mesh.faces][:, :, 2]
    is_flat_at_level = np.all(np.abs(face_z - z_level) < 1e-6, axis=1)
    path = mesh.outline(np.nonzero(is_flat_at_level)[0])
    return [path.vertices[entity.points][:, :2] for entity in path.entities]


def _is_hole_rim(contour, signed_area, min_area, max_area, max_absy):
    """Reject stray outlines: too few points, wrong size, or the logo band."""
    if len(contour) < 8:
        return False
    if not (min_area <= abs(signed_area) <= max_area):
        return False
    is_in_logo_band = abs(contour.mean(0)[1]) > max_absy
    return not is_in_logo_band


def outline(z=Z_TOP):
    """The board's outer rectangle at height z, as 4 corners in board metres.

    Excluded from the metrology fit -- 25 mm of transparent acrylic makes it a
    double edge up to 16 px apart at 4K -- but exact enough to DRAW, which is
    what makes board tracking checkable at a glance.
    """
    bounds = _mesh().bounds[:, :2] / 100.0     # STL is in centimetres
    (min_x, min_y), (max_x, max_y) = bounds[0], bounds[1]
    corners = np.array([[min_x, min_y], [max_x, min_y],
                        [max_x, max_y], [min_x, max_y]])
    return np.concatenate([corners, np.full((4, 1), z)], 1)


def pts3(pts2):
    return np.concatenate([pts2, np.full((len(pts2), 1), Z_TOP)], 1)


def build_levels(levels=((2.5, 0.8), (0.3, 0.35))):
    """Top rims plus pocket-floor rims: (pts3 m, nrm2, level, loop).

    The 22 mm baseline between the levels is the whole point. Top rims alone
    make the board one plane, whose tilt is visible only through perspective
    foreshortening -- weak over 320 mm at a 470 mm standoff, and strongly
    correlated with the along-normal translation. The floors turn tilt into a
    direct triangulation instead.
    """
    points, normals, level_ids, loop_ids = [], [], [], []
    loops_so_far = 0
    for level_id, (z_centimetres, min_area) in enumerate(levels):
        rim_xy, rim_normals, rim_loops = build(z_level=z_centimetres,
                                               min_area=min_area)
        z_metres = z_centimetres / 100.0
        points.append(np.concatenate(
            [rim_xy, np.full((len(rim_xy), 1), z_metres)], 1))
        normals.append(rim_normals)
        level_ids.append(np.full(len(rim_xy), level_id))
        # Loop ids restart at 0 per level; offset them so they stay unique.
        loop_ids.append(rim_loops + loops_so_far)
        loops_so_far += int(rim_loops.max()) + 1
    return (np.concatenate(points), np.concatenate(normals),
            np.concatenate(level_ids), np.concatenate(loop_ids))


# --------------------------------------------------------------- rim bias

def pad_bias(b):
    """Accept a shorter bias vector (an older bundle) and zero-fill the rest."""
    b = np.asarray(b, float).ravel()
    if len(b) >= NBIAS:
        return b[:NBIAS]
    return np.concatenate([b, np.zeros(NBIAS - len(b))])


def rim_bias(pts3, nrm2, level, R, t, bias):
    """Per-point radial offset of the rim contours, in metres.

    THE STL OUTLINE IS NOT WHERE THE INTENSITY EDGE IS. Measured over 144 views
    with the bundled poses, the signed rim residual is

        top rims   (z = 25 mm)   -0.733 px    model projects INSIDE the edge
        floor rims (z =  3 mm)   +0.306 px    model projects OUTSIDE the edge

    and the sign is the same in 144 of 144 views, so this is geometry, not
    noise. Both have an obvious cause: the top hole edge is chamfered (top face
    at 25 mm, shoulder at 23 mm) so the visible edge is not the z = 25 outline,
    and the floor rim sits at the base of a 22 mm wall whose shading gradient
    pulls the detected edge inward.

    Why it matters so much: tilt is triangulated from the 22 mm baseline between
    the two levels, so the DIFFERENTIAL bias between them is a tilt error by
    construction. 1.25 px of differential is 0.68 deg of tilt bias, and the
    view-to-view variation of that differential (sd 0.333 px) is 0.181 deg of
    tilt scatter -- which is the entire 0.182 deg per-view tilt error measured
    in the robot frame, i.e. 0.72 mm at the board corner.

    The model is, per level,

        delta = b0 + b1 * s + b2 * |s| + b3 * |u_xy|

    with u the unit view direction at the point, n its outward in-plane normal,
    and s = u_xy . n, which says whether the sight line runs into that piece of
    wall or across it.

    THE TERMS ARE NOT INTERCHANGEABLE, and the distinction is the whole point:

      b0  constant per level. Changes rim SIZE. Safe, and it carried the entire
          tilt fix -- see WHAT b0 IS WORTH.
      b1  linear in SIGNED s. Around a closed rim n sweeps every direction while
          u_xy is nearly constant, so this TRANSLATES the rim in the direction
          u_xy -- a pose parameter. Rejected; see WHY b1 IS ZERO.
      b2  linear in |s|. Symmetric about the rim, so it squashes the contour
          along the view azimuth without moving its centre.
      b3  linear in |u_xy| = sin(obliquity), a per-view scalar. A uniform
          per-view size change. Also symmetric.

    Measured, for a 1 mm coefficient on the top level at an oblique view:

        term   centroid shift   mean radial      what it does
        b0        0.00095 mm      1.0000 mm      pure size
        b1        0.10913        -0.0132         pure TRANSLATION  <- rejected
        b2        0.00674         0.1736         size, 26:1
        b3        0.00383         0.2700         size, 71:1

    b2 and b3 are therefore safe -- and they are also USELESS, which is a
    separate finding and the reason they default to zero. See WHY b2 AND b3
    ARE ZERO.

    The offsets are in METRES ON THE BOARD, not pixels, because that is what
    they measure as: the residual tracks distance with the sign and magnitude of
    a fixed board-plane offset (top +0.0015 px/mm, floor -0.0013 px/mm over a
    340-554 mm standoff, both shrinking toward zero as the camera backs off).
    A blur or PSF effect would have been fixed in pixels instead.

    WHAT b0 IS WORTH, bundled over 237 views:

        rim residual MAD    0.684 px  ->  0.265 px     (2.6x, for 2 parameters)
        b0_top   +0.1604 mm
        b0_floor -0.1124 mm
        differential        -0.2728 mm across the 22 mm baseline
                            = 0.71 deg of tilt the old model was carrying

    WHY b2 AND b3 ARE ZERO. They were built to catch the ~0.24 mm of
    reproducible, camera-orientation-dependent error that test_wrist_flex.py
    located and proved is not in the arm. They do not catch it. Fitted over 237
    views they take physically sensible values (b2_top +0.137, b2_floor -0.252,
    b3_top +0.090, b3_floor -0.102 mm -- nothing like b1's runaway) and they do
    improve the image fit. They just do not improve the POSE:

        metric                       b0          b0 + b2,b3
        rim residual MAD          0.265 px        0.250 px    better
        chain residual            0.762 mm        0.757 mm    flat
        leave-one-folder-out      0.645 mm        0.648 mm    worse
        multi-view split-half     0.208 mm        0.221 mm    worse, 7/11 folders

    Four parameters, a better picture, a worse answer. The pixel residual is not
    the objective and never was.

    What this rules out: the orientation-dependent error is NOT a rim radius or
    rim shape effect. It behaves like a genuine per-view POSE perturbation that
    happens to correlate with camera orientation, which points at the intrinsics
    or the arm rather than at the board model. Look there next, not here.

    WHY b1 IS ZERO. It is fittable, and fitting it does lower the pixel residual
    -- and it must not be used. b1 offsets each point by b1 * (u_xy . n), and
    around a closed rim n sweeps every direction while u_xy is nearly constant,
    so to first order that is a TRANSLATION of the rim in the direction u_xy.
    u_xy is nearly common to all rims in a view, so b1 is very nearly a per-view
    in-plane translation of the board -- a pose parameter. Fitted, it runs to
    1.74 and 2.11 mm, an order of magnitude larger than the entire effect being
    modelled, while improving the rim residual by 0.006 px over b0 alone
    (0.271 vs 0.265). That is the flex/droop failure mode again: a parameter
    absorbing pose, not measuring physics. bundle_rims.py --rim-bias b0b1 will
    still fit it, for anyone who wants to reproduce the finding.
    """
    camera_centre = -R.T @ t                       # board frame
    view_direction = camera_centre[None, :] - pts3
    view_direction /= np.maximum(
        np.linalg.norm(view_direction, axis=1, keepdims=True), 1e-12)

    # s says whether the sight line runs into this piece of wall or across it.
    into_wall = (view_direction[:, :2] * nrm2).sum(1)
    obliquity = np.hypot(view_direction[:, 0], view_direction[:, 1])

    b0, b1, b2, b3 = pad_bias(bias).reshape(4, 2)
    return (b0[level]
            + b1[level] * into_wall
            + b2[level] * np.abs(into_wall)
            + b3[level] * obliquity)


def biased_pts(pts3, nrm2, level, R, t, bias):
    """pts3 with each rim point pushed along its outward in-plane normal."""
    offset = rim_bias(pts3, nrm2, level, R, t, bias)
    biased = pts3.copy()
    biased[:, :2] += nrm2 * offset[:, None]
    return biased


def visible(pts3, nrm2, level, loop, R, t, z_shoulder=0.023):
    """Cull floor-rim points whose sight line is blocked by the pocket wall.

    The wall is vertical between the floor and the chamfer shoulder, so a floor
    point is visible iff the ray to the camera is still inside the pocket
    cross-section when it reaches the shoulder.
    """
    is_visible = np.ones(len(pts3), bool)
    camera_centre = -R.T @ t                       # board frame
    for level_id in np.unique(level):
        if level_id == 0:                          # top rims are never occluded
            continue
        for loop_id in np.unique(loop[level == level_id]):
            in_this_pocket = (level == level_id) & (loop == loop_id)
            floor_points = pts3[in_this_pocket]
            exit_xy, height_fraction = _ray_exit_at_shoulder(
                floor_points, camera_centre, z_shoulder)
            # The pocket's own floor outline IS its cross-section, so the ray
            # clears the wall iff it is still inside that outline at the
            # shoulder. A negative fraction means the shoulder is behind the
            # camera, which is not a sight line at all.
            clears_the_wall = _is_inside_polygon(exit_xy, floor_points[:, :2])
            is_visible[in_this_pocket] = clears_the_wall & (height_fraction > 0)
    return is_visible


def _ray_exit_at_shoulder(points, camera_centre, z_shoulder):
    """Where each point's sight line crosses the shoulder plane, and how far
    along the ray that is as a fraction of the distance to the camera."""
    to_camera = camera_centre[None, :] - points
    with np.errstate(divide="ignore", invalid="ignore"):
        height_fraction = (z_shoulder - points[:, 2]) / to_camera[:, 2]
    exit_xy = points[:, :2] + height_fraction[:, None] * to_camera[:, :2]
    return exit_xy, height_fraction


def _is_inside_polygon(query_xy, polygon):
    """Inside test for a convex polygon: on the inner side of every edge."""
    edge_start = polygon
    edge_vector = np.roll(polygon, -1, 0) - polygon
    inward = np.stack([edge_vector[:, 1], -edge_vector[:, 0]], 1)

    # The winding direction of the outline is not guaranteed, so orient the
    # normals using the centroid, which is inside by construction.
    centroid = polygon.mean(0)
    if np.mean(((centroid - edge_start) * inward).sum(1)) < 0:
        inward = -inward

    signed_distance = ((query_xy[:, None, :] - edge_start[None]) * inward[None]).sum(2)
    return np.all(signed_distance > 0, axis=1)


def _print_rim_summary():
    points, _, loop = build()
    rim_count = int(loop.max()) + 1
    print(f"{rim_count} rims, {len(points)} boundary points, "
          f"spacing {SPACING * 1000:.1f} mm")
    for rim_id in range(rim_count):
        rim = points[loop == rim_id]
        centre = rim.mean(0)
        radius = np.linalg.norm(rim - centre, axis=1)
        print(f"  rim {rim_id:2d}  n={len(rim):4d}  centre ({centre[0] * 1000:+8.2f},"
              f"{centre[1] * 1000:+8.2f}) mm   "
              f"r {radius.min() * 1000:5.2f}-{radius.max() * 1000:5.2f} mm")


if __name__ == "__main__":
    _print_rim_summary()
