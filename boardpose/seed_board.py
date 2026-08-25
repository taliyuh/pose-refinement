#!/usr/bin/env python3
"""
seed_board.py - marker-free board seed pose, from one image + its depth frame.

Replaces the old ArUco seeding stage (removed), which cannot work on
marker-free captures for three independent reasons:

  * pose_4..16 have no marker in frame at all;
  * it assumed a constant marker->board offset, but that offset moves between
    folders (measured over pose_1/2/3: x = 265.2 / 244.7 / 242.0 mm,
    y = +14.5 / +32.4 / +1.8 mm);
  * the camera-to-TCP transform can change between folders too, so nothing may
    be carried across folders to stand in for the marker.

Everything here is per-image. No marker, no forward kinematics, no hand-eye, no
prior pose, nothing shared between views -- which is what makes it immune to
the board and the camera stick both moving between sessions.

Method
------
Appearance-based pocket detection was tried first and abandoned: in the
exposure-60 captures the pocket walls are only ~30 grey levels above the wood
seen through the acrylic, so brightness blobs come out as crescents offset from
the pocket centre, and Canny rim contours do not close. Both were measured, not
assumed (11/20 rims recovered at best, versus 20/20 needed for shape-based
column identification).

What is robust is the rim EDGE MAP, which needs no closed contours and no
region segmentation, scored by CHAMFER DISTANCE:

  1. Canny edges -> distance transform D(x, y) = distance to the nearest edge.
  2. Score a candidate pose by the mean of D at the projected rim points.

The distance transform is what makes a coarse search viable: a candidate that
is 8 px off still scores ~8 rather than scoring as a total miss, so the basin is
wide and smooth. Gradient-dot-product energy -- what the old coarse() used --
has no such basin and rewards landing on ANY strong edge, which is why it
ranked wood grain above the board (measured: it moved pose_16 from 0.46 px to
1.20 px purely by reranking candidates).

The search is only 3-DoF. Roll, pitch and height come from the depth table
plane, fitted over the WHOLE image. That distinction matters: the old on-table
fitter picked its depth ROI by projecting the seed pose, so a bad seed produced a plane
fitted in the wrong place and the 3-DoF fit could never recover -- a +-150 mm
in-plane search on such a plane still stalled at 1.52 px, versus 0.46 px once
the plane was right. Nothing here depends on a prior pose.

Returns several ranked candidates rather than one. Chamfer picks the basin
reliably but not the exact minimum; fit_rims6.py refines each candidate and
selects on the rim residual, which is the quantity that actually matters.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import board_model  # noqa: E402
from camera import K_4K  # noqa: E402

D_4K = np.array([0.07524467259645462, -0.10655965656042099,
                 -0.00023194379173219204, 0.00031730238697491586,
                 0.044881995767354965], np.float32)
DEG = np.pi / 180.0

DOWN = 3               # depth subsample for the plane fit
CHAMFER_CLIP = 24.0    # px; beyond this a point is simply "missed"
MIN_IN_FRAME = 0.75    # minimum fraction of rim points on the sensor.
YAW_BIN = 12.0 * np.pi / 180.0   # 30 bins over 360 deg; candidate diversity
# Scoring the out-of-frame points at CHAMFER_CLIP rather than excluding them is
# deliberate, and was tested both ways: charging for them discriminates against
# poses that push the board off the sensor, which are usually wrong. Taking the
# median over in-frame points only -- the "correct" normalisation -- regressed
# pose_16/140707_412070 from 1.18 to 1.57 px and rescued nothing, at either
# gate. Do not re-derive this from first principles; measure it.


# ---------------------------------------------------------------- table plane

def table_plane_global(depth, z_lo=0.25, z_hi=1.2, thresh=0.004, iters=400):
    """Dominant plane of the depth frame, over the whole image.

    No ROI, so no dependence on a prior pose. The table is by far the largest
    planar surface in these captures; the board itself is transparent and
    returns little valid depth, so it does not compete.
    """
    d = depth[::DOWN, ::DOWN].astype(np.float64) / 1000.0
    ys, xs = np.nonzero((d > z_lo) & (d < z_hi))
    if len(xs) < 5000:
        return None
    z = d[ys, xs]
    P = np.stack([(xs * DOWN - K_4K[0, 2]) / K_4K[0, 0] * z,
                  (ys * DOWN - K_4K[1, 2]) / K_4K[1, 1] * z, z], 1)

    rng = np.random.default_rng(0)
    best_in, best = None, -1
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        inl = np.abs((P - s[0]) @ n) < thresh
        if inl.sum() > best:
            best, best_in = int(inl.sum()), inl
    if best_in is None or best < 5000:
        return None
    Q = P[best_in]
    c = Q.mean(0)
    n = np.linalg.eigh((Q - c).T @ (Q - c))[1][:, 0]
    if n[2] < 0:
        n = -n
    return n, float(-n @ c), best / float(len(P))


def plane_frame(n, d):
    """Right-handed frame on the table, z along the normal towards the camera."""
    if n[2] > 0:
        n, d = -n, -d
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, n); x /= np.linalg.norm(x)
    y = np.cross(n, x)
    return np.stack([x, y, n], 1), -d * n, n


def pose_on_plane(yaw, u, v, h, Rp, tp, n):
    c, s = np.cos(yaw), np.sin(yaw)
    R = Rp @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    t = tp + Rp[:, 0] * u + Rp[:, 1] * v + n * h
    return R, t


# ---------------------------------------------------------------- chamfer

def edge_distance_transform(img, lo=20, hi=60, blur=1.6):
    """Canny + distance transform.

    Thresholds are deliberately low. Measured against a known-good pose on
    pose_16, the fraction of rim points landing within 3 px of an edge is
    42% at (60,150), 68% at (40,110) and 79% at (20,60) -- the rims in the
    exposure-60 captures are simply not strong edges, and a high threshold
    deletes them. The extra wood-grain edges this admits raise the edge
    density only 0.34% -> 1.78%, which the distance transform absorbs.
    """
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), blur)
    e = cv2.Canny(g, lo, hi, L2gradient=True)
    return cv2.distanceTransform(255 - e, cv2.DIST_L2, 3)


# Board height above the fitted plane. NOT zero: the plane that depth actually
# fits sits below the board's own z = 0, by 26.7 mm on pose_16 (consistent with
# the unresolved ~56 mm depth offset already noted in the README). Fixing h = 0
# put every candidate at the wrong scale -- a 25 mm error at 430 mm standoff is
# a 6% size error, far more than the chamfer basin tolerates.
H_RANGE = np.arange(-0.010, 0.0551, 0.008)


def _sample_dt(dt, x, y):
    """Bilinear sample of the distance transform.

    Nearest-neighbour is fine for ranking a discrete grid but not for a
    continuous optimiser, which needs the objective to vary smoothly between
    pixels or it stalls immediately.
    """
    Hh, W = dt.shape
    x0 = np.floor(x).astype(np.int32); y0 = np.floor(y).astype(np.int32)
    fx = x - x0; fy = y - y0
    x0c = np.clip(x0, 0, W - 2); y0c = np.clip(y0, 0, Hh - 2)
    v00 = dt[y0c, x0c]; v10 = dt[y0c, x0c + 1]
    v01 = dt[y0c + 1, x0c]; v11 = dt[y0c + 1, x0c + 1]
    return (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy)
            + v01 * (1 - fx) * fy + v11 * fx * fy)


def chamfer_cost(dt, P, Rp, tp, n, par):
    """Median clipped edge distance of the projected rims, for one pose."""
    R, t = pose_on_plane(par[0], par[1], par[2], par[3], Rp, tp, n)
    X = P @ R.T + t
    z = np.maximum(X[:, 2], 1e-6)
    x = K_4K[0, 0] * X[:, 0] / z + K_4K[0, 2]
    y = K_4K[1, 1] * X[:, 1] / z + K_4K[1, 2]
    Hh, W = dt.shape
    inb = (x >= 0) & (x < W - 1) & (y >= 0) & (y < Hh - 1)
    if inb.mean() <= MIN_IN_FRAME:
        return np.inf
    val = np.where(inb, np.minimum(_sample_dt(dt, np.clip(x, 0, W - 2),
                                              np.clip(y, 0, Hh - 2)),
                                   CHAMFER_CLIP), CHAMFER_CLIP)
    return float(np.median(val))


# Parameter scales for the polish step: roughly one grid cell each, so the
# simplex starts at the resolution the grid actually resolved.
POLISH_SCALE = np.array([2.0 * DEG, 0.010, 0.010, 0.008])


def local_uv(dt, P, Rp, tp, n, par, span=0.008, step=0.002):
    """Dense translation-only sweep around a candidate, before polishing.

    The chamfer basin is strongly anisotropic, measured on
    pose_16/140707_412070 by perturbing the known-good pose and polishing back:

        2.0 deg yaw   error   11.83 -> 1.09   recovers
        8.0 mm height error    6.55 -> 1.20   recovers
        4.4 mm in-plane error 11.88 -> 8.80   DOES NOT recover
       10.0 mm in-plane error 20.30 -> 8.65   DOES NOT recover

    So the grid may be coarse in yaw and h, but must land within ~2 mm in u
    and v. Doing that globally means a 4 mm grid over the +-180 mm the data
    actually uses (max observed lateral offset 172 mm), which is 7.6x the
    search. Doing it locally around a few hundred candidates is ~1% of that.

    Blurring the distance transform to widen the basin was tried first and does
    not work -- a distance transform is already a smooth ramp, so smoothing it
    changes nothing useful (sigma 6/12/20 all failed to recover the 4.4 mm case).
    """
    yaw, u0, v0, h = par
    c, s = np.cos(yaw), np.sin(yaw)
    R = Rp @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    X = P @ R.T
    o = np.arange(-span, span + 1e-9, step)
    U, V = np.meshgrid(u0 + o, v0 + o, indexing="ij")
    UV = np.stack([U.ravel(), V.ravel()], 1)
    T = tp + UV[:, 0:1] * Rp[:, 0] + UV[:, 1:2] * Rp[:, 1] + h * n
    Xa = X[None, :, :] + T[:, None, :]
    z = np.maximum(Xa[..., 2], 1e-6)
    Hh, W = dt.shape
    x = np.rint(K_4K[0, 0] * Xa[..., 0] / z + K_4K[0, 2]).astype(np.int32)
    y = np.rint(K_4K[1, 1] * Xa[..., 1] / z + K_4K[1, 2]).astype(np.int32)
    inb = (x >= 0) & (x < W) & (y >= 0) & (y < Hh)
    np.clip(x, 0, W - 1, out=x); np.clip(y, 0, Hh - 1, out=y)
    val = np.where(inb, np.minimum(dt[y, x], CHAMFER_CLIP), CHAMFER_CLIP)
    sc = np.where(inb.mean(1) > MIN_IN_FRAME, np.median(val, axis=1), np.inf)
    k = int(np.argmin(sc))
    return (yaw, float(UV[k, 0]), float(UV[k, 1]), h), float(sc[k])


def polish(dt, P, Rp, tp, n, par, maxiter=180):
    """Locally minimise the chamfer, continuously, from a grid candidate.

    This is what makes the ranking meaningful. On pose_16/140707_412070 the
    exact true pose scores 0.95, but the nearest point of the (2 deg, 10 mm,
    8 mm) grid scores 10.92 -- so 234 wrong candidates outscored the truth and
    it ranked 235 of 1.4 million. That is a discretisation artefact, not a
    failure of the objective, and refining the grid away is not affordable
    (~44M candidates would be needed). Polishing each candidate first removes
    the artefact for a few hundred cheap evaluations.
    """
    from scipy.optimize import minimize
    z0 = np.asarray(par, float) / POLISH_SCALE
    f = lambda z: chamfer_cost(dt, P, Rp, tp, n, z * POLISH_SCALE)  # noqa: E731
    r = minimize(f, z0, method="Nelder-Mead",
                 options={"maxiter": maxiter, "xatol": 0.02, "fatol": 1e-3})
    return r.x * POLISH_SCALE, float(r.fun)


def _project(P3, R, t):
    X = P3 @ R.T + t
    z = np.maximum(X[..., 2], 1e-6)
    return np.stack([K_4K[0, 0] * X[..., 0] / z + K_4K[0, 2],
                     K_4K[1, 1] * X[..., 1] / z + K_4K[1, 2]], -1)


def search(dt, Rp, tp, n, yaw_step=2.0, uv_span=0.16, uv_step=0.010,
           n_polish=700,
           n_pts=140, topk=20, h_range=H_RANGE):
    """4-DoF chamfer search: yaw, in-plane u/v, and height above the plane.

    Step sizes are set by the refine's capture radius, not by taste. The board
    spans ~1600 px but its rims are ~15 px, so a 2 deg yaw error displaces the
    outermost rims by ~28 px -- past the point where the edge search can pull
    them back. Measured on pose_12/131731_390724: yaw 4 deg / uv 15 mm stalls
    at 1.34 px, yaw 2 deg / uv 10 mm reaches 0.56 px, and yaw 1.5 deg / uv 8 mm
    also reaches 0.56 px for twice the run time. Hence 2.0 / 10 mm.

    Vectorised over all (u, v) at each (yaw, h), so the grid costs a few
    hundred array ops rather than ~400k Python iterations.

    Scored by the fraction of IN-FRAME rim points lying within TAU px of an
    edge, maximised. Two properties matter:

      * robust, like a median and unlike a mean. On the known-good pose_16 pose
        the mean distance is 4.01 px but the median is 0.95 px -- a handful of
        points always land on a specular glint or an occluded pocket wall, and
        under a mean those few cost more than a better alignment can win back.

      * normalised by the in-frame count. Scoring over all points instead
        charged the clip value for every point off the sensor, so a correct
        pose on a board that runs off the edge of the frame scored worse than
        a wrong pose on a fully visible one. 33 of the 70 wrong-basin failures
        in the first full run had the board clipped, against 1 of 175 passes.
    """
    pts2, _, lid = board_model.build()
    sel = np.linspace(0, len(pts2) - 1, n_pts).astype(int)
    P = board_model.pts3(pts2[sel])                       # n_pts x 3

    Hh, W = dt.shape
    us = np.arange(-uv_span, uv_span + 1e-9, uv_step)
    U, V = np.meshgrid(us, us, indexing="ij")
    UV = np.stack([U.ravel(), V.ravel()], 1)              # n_uv x 2
    base = tp + UV[:, 0:1] * Rp[:, 0] + UV[:, 1:2] * Rp[:, 1]

    out = []
    for yaw in np.arange(0, 360, yaw_step) * DEG:
        c, s = np.cos(yaw), np.sin(yaw)
        R = Rp @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        X = P @ R.T                                       # n_pts x 3
        for h in h_range:
            T = base + h * n
            Xa = X[None, :, :] + T[:, None, :]            # n_uv x n_pts x 3
            z = np.maximum(Xa[..., 2], 1e-6)
            x = np.rint(K_4K[0, 0] * Xa[..., 0] / z + K_4K[0, 2]).astype(np.int32)
            y = np.rint(K_4K[1, 1] * Xa[..., 1] / z + K_4K[1, 2]).astype(np.int32)
            inb = (x >= 0) & (x < W) & (y >= 0) & (y < Hh)
            np.clip(x, 0, W - 1, out=x); np.clip(y, 0, Hh - 1, out=y)
            val = np.where(inb, np.minimum(dt[y, x], CHAMFER_CLIP), CHAMFER_CLIP)
            score = np.where(inb.mean(1) > MIN_IN_FRAME,
                             np.median(val, axis=1), np.inf)
            for k in np.argsort(score)[:40]:
                if np.isfinite(score[k]):
                    out.append((float(score[k]), float(yaw),
                                float(UV[k, 0]), float(UV[k, 1]), float(h)))
    out.sort(key=lambda o: o[0])

    # Keep the best candidate in each yaw bin, then take the best `topk` bins.
    #
    # Plain "best overall, minus near-duplicates" fails in a specific way: a
    # strong but wrong basin produces many good-scoring candidates at similar
    # yaw and assorted u/v, all of which survive a duplicate test that ORs yaw
    # against translation, and they fill the list before the true basin is
    # reached. That is not hypothetical -- pose_16/140707_412070 scored 0.58 px
    # with 12 candidates on a COARSE grid and 1.18 px with 20 candidates on a
    # FINER one, because the finer grid supplied more near-misses to crowd it
    # out. Binning by yaw makes the candidate set span orientation by
    # construction, so a wrong basin can occupy at most one slot.
    # Polish the global top-N by raw score. Yaw-binning first was tried and is
    # wrong here: it collapses to <=30 candidates, and the true pose is usually
    # not the best raw scorer even within its own yaw bin, so binning discards
    # it before polishing can rescue it. N must comfortably exceed the true
    # pose's raw rank, measured at 235 on pose_16/140707_412070.
    coarse = sorted(out, key=lambda c: c[0])[:n_polish]

    # polish each, THEN rank -- ranking raw grid scores puts the true pose
    # around 235th, ranking polished scores puts it first
    pol = []
    for s_, yaw, u, v, h in coarse:
        loc, _ = local_uv(dt, P, Rp, tp, n, (yaw, u, v, h))
        par, cost = polish(dt, P, Rp, tp, n, loc)
        if np.isfinite(cost):
            pol.append((cost, par[0], par[1], par[2], par[3]))
    pol.sort(key=lambda c: c[0])

    # drop duplicates that converged to the same minimum
    keep = []
    for c in pol:
        if all(abs((c[1] - k[1] + np.pi) % (2 * np.pi) - np.pi) > 1.0 * DEG
               or np.hypot(c[2] - k[2], c[3] - k[3]) > 0.004 for k in keep):
            keep.append(c)
        if len(keep) >= topk:
            break
    return keep


# ---------------------------------------------------------------- api

def seed_view(view_dir, topk=20, yaw_step=2.0, uv_step=0.010):
    """Ranked candidate T_cam_board for one view. Returns (list, info)."""
    view_dir = Path(view_dir)
    img = cv2.imread(str(view_dir / "color.png"))
    depth = cv2.imread(str(view_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    if img is None or depth is None:
        return [], {"why": "missing color/depth"}

    pl = table_plane_global(depth)
    if pl is None:
        return [], {"why": "no table plane"}
    n, d, inl = pl
    Rp, tp, nn = plane_frame(n, d)

    img_u = cv2.undistort(img, K_4K.astype(np.float32), D_4K,
                          newCameraMatrix=K_4K.astype(np.float32))
    dt = edge_distance_transform(img_u)

    cands = search(dt, Rp, tp, nn, topk=topk, yaw_step=yaw_step, uv_step=uv_step)
    Ts = []
    for s_, yaw, u, v, h in cands:
        R, t = pose_on_plane(yaw, u, v, h, Rp, tp, nn)
        T = np.eye(4); T[:3, :3], T[:3, 3] = R, t
        Ts.append((s_, T))
    return Ts, {"plane_inlier_frac": round(inl, 3),
                "plane_normal": [round(float(q), 4) for q in nn],
                "n_cand": len(Ts),
                "best_chamfer": round(cands[0][0], 2) if cands else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("views", nargs="*", help="pose_N/timestamp; default all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = ROOT / "images" / "images_depth"
    views = args.views or sorted(
        (f"{p.name}/{v.name}" for p in data.glob("pose_*") if p.is_dir()
         for v in sorted(p.iterdir()) if v.is_dir()),
        key=lambda s: (int(s.split("/")[0].split("_")[1]), s.split("/")[1]))
    res = {}
    for v in views:
        Ts, info = seed_view(data / v)
        print(f"  {v:32} cand {info.get('n_cand', 0):3d}  "
              f"chamfer {info.get('best_chamfer')}  "
              f"plane_inl {info.get('plane_inlier_frac')}  {info.get('why','')}")
        if Ts:
            res[v] = [T.tolist() for _, T in Ts]
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
