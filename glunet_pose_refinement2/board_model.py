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


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _resample(p, spacing):
    """Uniform arc-length resampling of a closed polygon."""
    q = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(24, int(round(s[-1] / spacing)))
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])], 1)


_MESH = None
_CACHE = {}


def _mesh():
    """The STL, parsed once per process.

    build() used to re-parse it on every call, and build_levels() calls build()
    twice -- so a per-view fit paid for three trimesh loads plus three outline
    extractions, 753 redundant parses over the 251-view run. Measured cost is
    128 ms per parse, i.e. ~0.4 s of the 14.5 s per view (~3%); the run is
    dominated by the chamfer search and the candidate refines, not by this.
    Cached anyway because it is free to do.
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
    m = _mesh()
    v, f = m.vertices, m.faces
    face_on = np.all(np.abs(v[f][:, :, 2] - z_level) < 1e-6, axis=1)
    path = m.outline(np.nonzero(face_on)[0])

    pts, nrm, lid = [], [], []
    for e in path.entities:
        p = path.vertices[e.points][:, :2]
        if len(p) < 8:
            continue
        A = _area(p)
        if not (min_area <= abs(A) <= max_area):
            continue
        if abs(p.mean(0)[1]) > max_absy:     # drop the logo/text band
            continue
        q = _resample(p, SPACING * 100.0) / 100.0     # cm -> m
        tan = np.roll(q, -1, 0) - np.roll(q, 1, 0)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True)
        nn = np.stack([tan[:, 1], -tan[:, 0]], 1)     # right-hand perpendicular
        if A > 0:                                     # CCW -> flip to outward
            nn = -nn
        k = len(pts)
        pts.append(q); nrm.append(nn); lid.append(np.full(len(q), k))
    if not pts:
        raise RuntimeError("no rims found")
    out = (np.concatenate(pts), np.concatenate(nrm), np.concatenate(lid))
    _CACHE[key] = out
    return tuple(a.copy() for a in out)


def outline(z=Z_TOP):
    """The board's outer rectangle at height z, as 4 corners in board metres.

    Excluded from the metrology fit -- 25 mm of transparent acrylic makes it a
    double edge up to 16 px apart at 4K -- but exact enough to DRAW, which is
    what makes board tracking checkable at a glance.
    """
    b = _mesh().bounds[:, :2] / 100.0          # STL is in centimetres
    (x0, y0), (x1, y1) = b[0], b[1]
    q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return np.concatenate([q, np.full((4, 1), z)], 1)


def pts3(pts2):
    return np.concatenate([pts2, np.full((len(pts2), 1), Z_TOP)], 1)


if __name__ == "__main__":
    p, n, l = build()
    print(f"{l.max() + 1} rims, {len(p)} boundary points, "
          f"spacing {SPACING * 1000:.1f} mm")
    for k in range(int(l.max()) + 1):
        q = p[l == k]
        c = q.mean(0)
        r = np.linalg.norm(q - c, axis=1)
        print(f"  rim {k:2d}  n={len(q):4d}  centre ({c[0] * 1000:+8.2f},"
              f"{c[1] * 1000:+8.2f}) mm   r {r.min() * 1000:5.2f}-{r.max() * 1000:5.2f} mm")


def build_levels(levels=((2.5, 0.8), (0.3, 0.35))):
    """Multi-level contours: the top rims AND the pocket floor rims.

    The pockets are blind, 22 mm deep, with floors at z = 3 mm. Fitting only
    the top rims makes the board a single plane, and a plane's out-of-plane
    tilt is observable solely through perspective foreshortening -- weak at
    320 mm across a 470 mm standoff, and strongly correlated with the
    along-normal translation. The floor rims sit 22 mm below the top rims, so
    including them turns tilt into a direct triangulation with a real baseline.

    Returns (pts3 Nx3 metres, nrm2 Nx2 in-plane outward normals, level, loop).
    """
    P, N, L, G, base = [], [], [], [], 0
    for li, (zc, min_area) in enumerate(levels):
        p, n, g = build(z_level=zc, min_area=min_area)
        z = zc / 100.0
        P.append(np.concatenate([p, np.full((len(p), 1), z)], 1))
        N.append(n); L.append(np.full(len(p), li)); G.append(g + base)
        base += int(g.max()) + 1
    return (np.concatenate(P), np.concatenate(N),
            np.concatenate(L), np.concatenate(G))


# --------------------------------------------------------------- rim bias

RIM_BIAS_NAMES = ["b0_top", "b0_floor", "b1_top", "b1_floor",
                  "b2_top", "b2_floor", "b3_top", "b3_floor"]
NBIAS = len(RIM_BIAS_NAMES)

# Calibrated by bundle_rims.py; see rim_bias(). Units: metres of offset along
# each rim's outward in-plane normal. b1 is held at zero on purpose -- it is
# fittable and must not be fitted, see WHY b1 IS ZERO.
RIM_BIAS0 = np.array([1.604e-4, -1.124e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


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
    C = -R.T @ t                                   # camera centre, board frame
    u = C[None, :] - pts3
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    s = (u[:, :2] * nrm2).sum(1)
    uxy = np.hypot(u[:, 0], u[:, 1])
    b = pad_bias(bias)
    return (b[0:2][level] + b[2:4][level] * s
            + b[4:6][level] * np.abs(s) + b[6:8][level] * uxy)


def biased_pts(pts3, nrm2, level, R, t, bias):
    """pts3 with each rim point pushed along its outward in-plane normal."""
    d = rim_bias(pts3, nrm2, level, R, t, bias)
    q = pts3.copy()
    q[:, :2] += nrm2 * d[:, None]
    return q


def visible(pts3, nrm2, level, loop, R, t, z_shoulder=0.023):
    """Cull floor-rim points whose sight line is blocked by the pocket wall.

    The wall is vertical between the floor and the chamfer shoulder, so a floor
    point is visible iff the ray to the camera is still inside the pocket
    cross-section when it reaches the shoulder.
    """
    ok = np.ones(len(pts3), bool)
    C = -R.T @ t                                   # camera centre, board frame
    for li in np.unique(level):
        if li == 0:
            continue
        for g in np.unique(loop[level == li]):
            m = (level == li) & (loop == g)
            p = pts3[m]
            d = C[None, :] - p
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (z_shoulder - p[:, 2]) / d[:, 2]
            exit_xy = p[:, :2] + s[:, None] * d[:, :2]
            # inside test against the pocket outline (its own xy polygon)
            poly = p[:, :2]
            c = poly.mean(0)
            # a point is inside iff it is on the inner side of every edge
            a, b = poly, np.roll(poly, -1, 0)
            e = b - a
            nrm = np.stack([e[:, 1], -e[:, 0]], 1)
            if np.mean(((c - a) * nrm).sum(1)) < 0:
                nrm = -nrm
            inside = np.all(((exit_xy[:, None, :] - a[None]) * nrm[None]).sum(2) > 0, axis=1)
            ok[m] = inside & (s > 0)
    return ok
