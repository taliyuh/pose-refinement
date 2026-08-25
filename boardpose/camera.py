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


def scale_K(K, from_wh, to_wh):
    """Rescale intrinsics between two image sizes of the same aspect ratio."""
    sx, sy = to_wh[0] / from_wh[0], to_wh[1] / from_wh[1]
    out = np.array(K, np.float64)
    out[0, :] *= sx
    out[1, :] *= sy
    return out


# ---------------------------------------------------------------- edge search

def sample(img, x, y):
    """Bilinear sample of `img` at float coordinates."""
    return cv2.remap(img, x.astype(np.float32), y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def edge_search(gx, gy, pm, d, S):
    """Subpixel signed offset from each projected rim point to the strongest
    edge along its normal.

    gx, gy  image gradients (float32)
    pm      (N,2) projected rim points
    d       (N,2) unit image-space normals to search along
    S       search half-width in pixels

    Returns (offset, strength, valid). The offset is signed along `d`; the
    parabola fit through the gradient-magnitude peak is what makes it subpixel,
    and `valid` is False where the peak sat on the edge of the search window
    (i.e. the true edge is probably outside it).
    """
    offs = np.arange(-S, S + 1, dtype=np.float64)
    sx = pm[:, 0][:, None] + d[:, 0][:, None] * offs[None, :]
    sy = pm[:, 1][:, None] + d[:, 1][:, None] * offs[None, :]
    dd = (sample(gx, sx, sy) * d[:, 0][:, None]
          + sample(gy, sx, sy) * d[:, 1][:, None])
    a = np.abs(dd)
    k = a.argmax(1)
    r = np.arange(len(k))
    peak = a[r, k]
    ok = (k > 0) & (k < len(offs) - 1) & (peak > 0)
    ki = np.clip(k, 1, len(offs) - 2)
    y0, y1, y2 = a[r, ki - 1], a[r, ki], a[r, ki + 1]
    den = y0 - 2 * y1 + y2
    safe = np.abs(den) > 1e-9
    sub = np.zeros_like(den)
    np.divide(0.5 * (y0 - y2), den, out=sub, where=safe)
    return offs[ki] + np.clip(sub, -1, 1), peak, ok
