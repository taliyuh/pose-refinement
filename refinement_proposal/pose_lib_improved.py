#!/usr/bin/env python3
"""
Improved shared core for transparent-board 3-DOF pose estimation.
Uses exact 3D centers of shapes from CAD coordinates.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt

# add workspace to path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# downsize the image otherwise it takes too long
IW, IH = 1280, 720

# large gaussian blur window to eliminate uneven lighting
FLATTEN_KERNEL = 101

def load_gray(path):

    # read the image
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)

    # resize to defined scale
    img = cv2.resize(img, (IW, IH))

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # return both
    return img, gray

def flatten_illumination(gray):

    # blur the image to remove illumination gradient
    blur = cv2.GaussianBlur(gray, (FLATTEN_KERNEL, FLATTEN_KERNEL), 0)
    blur = np.clip(blur, 1, None).astype(np.float32)
    flat = (gray.astype(np.float32) / blur) * 128.0
    return np.clip(flat, 0, 255).astype(np.uint8)

# camera intrinsics
K = np.array([[750.6503295898438, 0.0, 643.8098754882812],
              [0.0, 750.5856323242188, 363.8945617675781],
              [0.0, 0.0, 1.0]], dtype=np.float64)

# camera distortion
DIST = np.zeros((4, 1))

# downscale factor for the image
SW, SH = 3840, 2160
S = np.array([IW / SW, IH / SH])

# board corners (m), top face z = 2.5 cm
# read from the .stl file
CORNER_ORDER = ["top left", "top right", "bottom left", "bottom right"]
CORNER_3D = np.float32([[-0.16, 0.125, 0.025], [0.16, 0.125, 0.025],
                        [-0.16, -0.125, 0.025], [0.16, -0.125, 0.025]])

# centres of cutours from the board, also from the .stl
MODEL_CENTERS = np.float32([
    # circles
    [-0.12784,  0.0650, 0.025], [-0.12784,  0.0125, 0.025],
    [-0.12784, -0.03865, 0.025], [-0.12784, -0.09503, 0.025],
    # hexagons
    [-0.067600, 0.065000, 0.025], [-0.067663, 0.012522, 0.025],
    [-0.067563, -0.038596, 0.025], [-0.066478, -0.094928, 0.025],
    # triangles
    [-0.00004369, 0.064959, 0.025], [-0.0000591, 0.012542, 0.025],
    [0.0, -0.03865, 0.025], [-0.0000336, -0.095106, 0.025],
    # rectangles
    [0.060873, 0.064935, 0.025], [0.060872, 0.012594, 0.025],
    [0.060911, -0.03865, 0.025], [0.060911, -0.095033, 0.025]
])

# select cylinders
CYL_IDX = np.array([0, 1, 2, 3])

# other shapes
_OTHER_IDX = np.array([i for i in range(len(MODEL_CENTERS)) if i not in CYL_IDX])


def solve_corners(corners_720):
    # map 3d coordinates of the board to corresponding 2d points detected
    _, rv, tv = cv2.solvePnP(CORNER_3D, np.float32(corners_720), K, DIST,
                             flags=cv2.SOLVEPNP_ITERATIVE)
    return rv, tv


def pose_in_plane(params, R_plane, t_plane):
    th, u, v = params
    Rz = np.array([[np.cos(th), -np.sin(th), 0],
                   [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    R = R_plane @ Rz
    t = np.asarray(t_plane).ravel() + R_plane @ np.array([u, v, 0.0])
    return cv2.Rodrigues(R)[0], t.reshape(3, 1)


def project(p3, rv, tv):
    # projects 3d points to 2d using camera params
    return cv2.projectPoints(p3, rv, tv, K, DIST)[0].reshape(-1, 2)


def detect_blobs(gray, roi, element_size=35, thr=60, return_meta=False):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (element_size, element_size))

    # extract bright elements smaller than the structuring element size
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)

    # stretch the image contrast so that the features stand out brightly
    th = cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # convert to binary
    _, b = cv2.threshold(th, thr, 255, cv2.THRESH_BINARY)

    # apply roi mask to ignore aruco marker
    if roi is not None:
        b = cv2.bitwise_and(b, b, mask=roi)

    # remove small noise
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # find contours
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs, iscyl, areas, circs, radii = [], [], [], [], []

    for c in contours:

        # filter contours by area and ratio of area to convex hull to reject noise
        a = cv2.contourArea(c)
        if a < 40 or a > 2500:
            continue
        if a / max(cv2.contourArea(cv2.convexHull(c)), 1) < 0.75:
            continue

        # measure circularity - perfect circle returns 1.0
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * a / (peri * peri) if peri else 0

        # compute centroid of contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        blobs.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])

        # keep track of circles
        iscyl.append(circ > 0.78)
        areas.append(a)
        circs.append(circ)
        radii.append(np.sqrt(a / np.pi))

    # format variables to float and boolean numpy arrays
    blobs = np.float32(blobs).reshape(-1, 2)
    iscyl = np.array(iscyl, bool)
    if return_meta:
        meta = {"area": np.array(areas), "circ": np.array(circs),
                "radius": np.array(radii)}
        return blobs, iscyl, meta
    return blobs, iscyl


def detect_cylinder_line(blobs, min_pts=3, tol=7.0, even_tol=0.45):

    # search the set of 2d blobs to find a subset that aligns on a line
    n = len(blobs)
    if n < min_pts:
        return None

    # score lines, keep the best one
    best = None
    for i in range(n):
        for j in range(i + 1, n):

            # calculate vector and length of segment
            d = blobs[j] - blobs[i]
            L = np.hypot(*d)

            # reject pairs that are too close or too far apart
            if L < 25 or L > 420:
                continue

            # normalize
            nrm = np.array([-d[1], d[0]]) / L

            # points on either side of the line
            perp = np.abs((blobs - blobs[i]) @ nrm)
            inl = np.where(perp < tol)[0]
            if len(inl) < min_pts:
                continue

            # check if gaps are even
            t = np.sort((blobs[inl] - blobs[i]) @ (d / L))
            gaps = np.diff(t)
            if gaps.std() / max(gaps.mean(), 1e-6) > even_tol:
                continue

            # keep track of inliers, sort points by distance from the start of the line
            order = inl[np.argsort((blobs[inl] - blobs[i]) @ (d / L))]
            score = len(inl) * 1000 - gaps.std()
            if best is None or score > best[0]:
                best = (score, blobs[order])
    
    # return points sorted along the line
    return None if best is None else best[1]


def cost_fn(blobs, iscyl, R_plane, t_plane, gate=32.0, cyl_w=2.0, min_inl=6):

    # measure how well the candidate pose matches the image detected blobs
    cyl_blobs = blobs
    oth_blobs = blobs
    cyl_model = MODEL_CENTERS[CYL_IDX]
    oth_model = MODEL_CENTERS[_OTHER_IDX]


    # use hungarian algorithm to find best matching blobs to model centers
    def _assign(pm, pb, g):
        m = len(pm)
        if m == 0:
            return 0.0, 0
        if len(pb) == 0:
            return g * m, 0
        D = np.minimum(cdist(pm, pb), g)
        row, col = linear_sum_assignment(D)
        matched = D[row, col] < g
        n_in = int(matched.sum())
        total = float(D[row, col][matched].sum() + g * (m - n_in))
        return total, n_in

    # compute cost function - takes into account cylinders and other shapes
    denom = cyl_w * len(cyl_model) + len(oth_model)

    # compute cost function
    def cost(params):
        # get rotation and translation from params
        rv, tv = pose_in_plane(params, R_plane, t_plane)
        
        # project model centers to 2d
        pc = project(cyl_model, rv, tv)
        po = project(oth_model, rv, tv)
        p = np.vstack([pc, po])

        # check if points are within image bounds
        vis = (p[:, 0] >= 0) & (p[:, 0] < IW) & (p[:, 1] >= 0) & (p[:, 1] < IH)
        if vis.sum() < min_inl:
            return 1e6
        
        # assign blobs to model centers
        cc, _ = _assign(pc, cyl_blobs, gate)
        oc, _ = _assign(po, oth_blobs, gate)
        return (cyl_w * cc + oc) / denom

    def count_inliers(params):
        rv, tv = pose_in_plane(params, R_plane, t_plane)
        _, cin = _assign(project(cyl_model, rv, tv), cyl_blobs, gate)
        _, oin = _assign(project(oth_model, rv, tv), oth_blobs, gate)
        return cin + oin

    cost.count_inliers = count_inliers
    cost.min_inl = min_inl
    return cost


_CYL_C = MODEL_CENTERS[CYL_IDX].mean(axis=0)
_OTH_C = MODEL_CENTERS[_OTHER_IDX].mean(axis=0)


def cylinder_pose_seed(blobs, R_plane, t_plane):
    line = detect_cylinder_line(blobs)
    if line is None or len(line) < 4:
        return None
    lineset = {(round(x, 1), round(y, 1)) for x, y in line}
    others = np.array([b for b in blobs
                       if (round(b[0], 1), round(b[1], 1)) not in lineset])
    if len(others) < 3:
        return None
    mid = line.mean(axis=0)
    d = line[-1] - line[0]
    nrm = np.array([-d[1], d[0]]) / np.hypot(*d)
    lang = np.arctan2(d[1], d[0])
    side_obs = np.sign((others.mean(axis=0) - mid) @ nrm)

    u_mid, v_mid = backproject_to_plane(mid, R_plane, t_plane)
    best = None
    for th in np.linspace(-np.pi, np.pi, 360, endpoint=False):
        Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        off = Rz @ _CYL_C[:2]
        uu, vv = u_mid - off[0], v_mid - off[1]
        rv, tv = pose_in_plane((th, uu, vv), R_plane, t_plane)
        pc = project(MODEL_CENTERS[CYL_IDX], rv, tv)
        poc = project(_OTH_C.reshape(1, 3), rv, tv)[0]
        pd = pc[-1] - pc[0]
        pd_angle = np.arctan2(pd[1], pd[0])
        diff = pd_angle - lang
        aerr = abs(np.arctan2(np.sin(2 * diff), np.cos(2 * diff))) / 2.0
        side_proj = np.sign((poc - pc.mean(axis=0)) @ nrm)
        serr = 0.0 if side_proj == side_obs else np.pi
        e = aerr + serr
        if best is None or e < best[0]:
            best = (e, (th, uu, vv))
    return best[1]


def backproject_to_plane(px, R_plane, t_plane, z_local=0.025):
    d = np.linalg.inv(K) @ np.array([px[0], px[1], 1.0])
    n = R_plane[:, 2]
    P0 = np.asarray(t_plane).ravel() + z_local * n
    s = float(n @ P0) / float(n @ d)
    X = s * d
    uvw = R_plane.T @ (X - np.asarray(t_plane).ravel())
    return float(uvw[0]), float(uvw[1])


def estimate(blobs, iscyl, R_plane, t_plane, n_th=72, n_uv=15, uv=0.15):
    """3-DOF fit."""
    coarse = cost_fn(blobs, iscyl, R_plane, t_plane, gate=48.0)
    fine = cost_fn(blobs, iscyl, R_plane, t_plane, gate=22.0)

    if len(blobs):
        u0, v0 = backproject_to_plane(blobs.mean(axis=0), R_plane, t_plane)
    else:
        u0, v0 = 0.0, 0.0

    th_list = list(np.linspace(-np.pi, np.pi, n_th, endpoint=False))
    # Restore win = 0.13 for fallback coarse grid search
    win = 0.13
    us = u0 + np.linspace(-win, win, n_uv)
    vs = v0 + np.linspace(-win, win, n_uv)

    per_yaw = []
    for th in th_list:
        b, bc = None, 1e9
        for u in us:
            for v in vs:
                c = coarse((th, u, v))
                if c < bc:
                    bc, b = c, (th, u, v)
        if b is not None:
            per_yaw.append((bc, b))
    per_yaw.sort(key=lambda z: z[0])
    starts = [b for _, b in per_yaw[:8]]

    cyl = cylinder_pose_seed(blobs, R_plane, t_plane)
    if cyl is not None:
        th_c, uc, vc = cyl
        starts = [(th_c, uc, vc), (th_c + np.pi, uc, vc)] + starts
    else:
        if not starts:
            starts = [(0.0, u0, v0)]

    # refine each start with the tight gate, then pick the global best
    dth = 2 * np.pi / n_th
    duv = 2 * win / (n_uv - 1)
    best, bc = None, 1e9
    for st in starts:
        b, c = st, fine(st)
        for th in b[0] + np.linspace(-dth, dth, 7):
            for u in b[1] + np.linspace(-duv, duv, 7):
                for v in b[2] + np.linspace(-duv, duv, 7):
                    cc = fine((th, u, v))
                    if cc < c:
                        c, b = cc, (th, u, v)
        res = minimize(fine, np.array(b), method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 600})
        if res.fun < c:
            b, c = tuple(res.x), res.fun
        if c < bc:
            bc, best = c, b
    rv, tv = pose_in_plane(best, R_plane, t_plane)
    return np.asarray(best), float(bc), rv, tv
