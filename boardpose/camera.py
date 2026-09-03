#!/usr/bin/env python3
"""
camera.py - camera intrinsics and the subpixel edge search.

Everything in the pipeline that touches pixels needs these three things: the 4K
intrinsics, the distortion coefficients, and `edge_search`. They used to live in
`tilt_vs_depth.py` and `fit_dense_on_table.py` -- two superseded scripts that
were kept alive purely because the live code imported constants out of them, and
which in turn dragged the whole legacy ArUco stack into every import. Collected
here so the dependency is a leaf.

INTRINSICS ARE A PROPERTY OF YOUR CAMERA, NOT OF THIS CODE. The values below
are the factory intrinsics of the Orbbec Femto used for the reference dataset,
at 3840x2160. They are only a starting point: `bundle_rims.py` re-estimates
fx, fy, cx, cy, k1, k2, p1, p2, k3 across all views and writes the refined set
into its bundle, and every downstream consumer reads them from there. So a
different camera needs these replaced only well enough for the initial fit to
converge -- take them from `camera_info` (ROS) or any checkerboard calibration.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------- intrinsics

# Orbbec Femto CL8M84100BH, 3840x2160, as reported by /camera_info.
# Cross-checked by the bundle, which fitted fx = 2251.4 against 2252.0 across
# 47 views spanning 402-628 mm of standoff -- a -0.02% shift with cx/cy unmoved.
K_4K = np.array([[2251.950927734375, 0, 1931.4296875],
                 [0, 2251.7568359375, 1091.6837158203125],
                 [0, 0, 1]], np.float64)

D_4K = np.array([0.07524467259645462, -0.10655965656042099,
                 -0.00023194379173219204, 0.00031730238697491586,
                 0.044881995767354965], np.float32)

# Below this curvature the parabola through the three peak samples is flat
# enough that its vertex is numerically meaningless.
_MIN_PARABOLA_CURVATURE = 1e-9


def scale_K(K, from_width_height, to_width_height):
    """Rescale intrinsics between two image sizes of the same aspect ratio."""
    scale_x = to_width_height[0] / from_width_height[0]
    scale_y = to_width_height[1] / from_width_height[1]
    scaled = np.array(K, np.float64)
    scaled[0, :] *= scale_x
    scaled[1, :] *= scale_y
    return scaled


def sample(image, x, y):
    """Bilinear sample of `image` at float coordinates."""
    return cv2.remap(image, x.astype(np.float32), y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _directional_gradient_profile(grad_x, grad_y, points, normals, offsets):
    """Gradient projected onto each point's normal, sampled along that normal."""
    sample_x = points[:, 0][:, None] + normals[:, 0][:, None] * offsets[None, :]
    sample_y = points[:, 1][:, None] + normals[:, 1][:, None] * offsets[None, :]
    return (sample(grad_x, sample_x, sample_y) * normals[:, 0][:, None]
            + sample(grad_y, sample_x, sample_y) * normals[:, 1][:, None])


def _parabola_vertex_offset(left, centre, right):
    """Vertex of the parabola through three unit-spaced samples, in samples."""
    curvature = left - 2 * centre + right
    vertex = np.zeros_like(curvature)
    is_curved = np.abs(curvature) > _MIN_PARABOLA_CURVATURE
    np.divide(0.5 * (left - right), curvature, out=vertex, where=is_curved)
    # A vertex beyond one sample either way means the peak is not really here.
    return np.clip(vertex, -1, 1)


def edge_search(grad_x, grad_y, points, normals, search_halfwidth):
    """Subpixel signed offset from each rim point to the strongest edge, in px.

    Returns (offset along `normals`, peak strength, valid). `valid` is False
    where the peak sat on the window edge, i.e. the true edge is outside it.
    """
    offsets = np.arange(-search_halfwidth, search_halfwidth + 1, dtype=np.float64)
    magnitude = np.abs(_directional_gradient_profile(
        grad_x, grad_y, points, normals, offsets))

    peak_index = magnitude.argmax(1)
    point_index = np.arange(len(peak_index))
    peak_strength = magnitude[point_index, peak_index]

    # A peak on either end of the window means the true edge is probably
    # outside it, so the parabola has no bracketing sample to fit through.
    is_interior_peak = ((peak_index > 0) & (peak_index < len(offsets) - 1)
                        & (peak_strength > 0))

    clamped = np.clip(peak_index, 1, len(offsets) - 2)
    subpixel = _parabola_vertex_offset(magnitude[point_index, clamped - 1],
                                       magnitude[point_index, clamped],
                                       magnitude[point_index, clamped + 1])
    return offsets[clamped] + subpixel, peak_strength, is_interior_peak
