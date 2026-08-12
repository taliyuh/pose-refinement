#!/usr/bin/env python3
"""
fit_dense_on_table.py - dense, on-table board pose fit.

Replaces the 4-corner PnP bottleneck. Two changes:

  1. Constrain the pose to the table.  The board lies flat, so roll, pitch and
     height are not estimated from the image at all -- they come from the depth
     camera's table plane (a 60k-point fit, split-half stable to ~1.5 deg).
     That removes the DoF the 4-corner solve gets catastrophically wrong
     (measured: median 1.3 deg, up to 51 deg, invisible in the mask).

  2. Fit ~1200 hole-rim boundary points from the STL instead of 4 extrapolated
     corners, by direct subpixel edge search along each rim normal. No feature
     detector, no GLU-Net at inference -- GLU-Net only seeds the search.

Free parameters: yaw, u, v  (3-DoF, in the table plane), optionally h.

The fit is over-determined ~400:1, so its residual is a real measurement of
agreement rather than being identically zero as it is for 4-point PnP. That
residual is the quality gate.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import board_model  # noqa: E402
from tilt_vs_depth import K_4K, S as SCALE, table_plane  # noqa: E402

D_4K = np.array([0.07524467259645462, -0.10655965656042099,
                 -0.00023194379173219204, 0.00031730238697491586,
                 0.044881995767354965], np.float32)
DEG = np.pi / 180.0


# ---------------------------------------------------------------- geometry

def plane_frame(n, d):
    """Right-handed frame on the table: z = n (towards the camera)."""
    if n[2] > 0:
        n, d = -n, -d
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, n); x /= np.linalg.norm(x)
    y = np.cross(n, x)
    return np.stack([x, y, n], 1), -d * n, n


def pose(params, Rp, tp, n):
    yaw, u, v, h = params
    c, s = np.cos(yaw), np.sin(yaw)
    R = Rp @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    t = tp + Rp[:, 0] * u + Rp[:, 1] * v + n * h
    return R, t


def project(P3, R, t):
    X = P3 @ R.T + t
    z = np.maximum(X[:, 2], 1e-6)
    return np.stack([K_4K[0, 0] * X[:, 0] / z + K_4K[0, 2],
                     K_4K[1, 1] * X[:, 1] / z + K_4K[1, 2]], 1)


def project_with_normals(pts2, nrm2, params, Rp, tp, n):
    """Projected rim points and the unit image-space direction of their normals."""
    R, t = pose(params, Rp, tp, n)
    P = board_model.pts3(pts2)
    Q = board_model.pts3(pts2 + nrm2 * 1e-3)      # 1 mm along the outward normal
    a, b = project(P, R, t), project(Q, R, t)
    d = b - a
    L = np.linalg.norm(d, axis=1, keepdims=True)
    return a, d / np.maximum(L, 1e-9)


# ---------------------------------------------------------------- image

def sample(img, x, y):
    return cv2.remap(img, x.astype(np.float32), y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def edge_search(gx, gy, pm, d, S):
    """Subpixel signed offset from each projected rim point to the strongest
    edge along its normal. Returns (offset, strength, valid)."""
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


# ---------------------------------------------------------------- fitting

def coarse(gx, gy, pts2, nrm2, Rp, tp, n, init, yaw_span, yaw_step,
           uv_span, uv_step, sub=4):
    """Grid search maximising edge energy under the projected rim outline."""
    p2, n2 = pts2[::sub], nrm2[::sub]
    best, bestv = init, -np.inf
    yaws = init[0] + np.arange(-yaw_span, yaw_span + 1e-9, yaw_step) * DEG
    uvs = np.arange(-uv_span, uv_span + 1e-9, uv_step)
    for q in (0, 90, 180, 270):
        for yaw in yaws + q * DEG:
            for du in uvs:
                for dv in uvs:
                    par = (yaw, init[1] + du, init[2] + dv, init[3])
                    a, d = project_with_normals(p2, n2, par, Rp, tp, n)
                    dd = (sample(gx, a[:, 0][:, None], a[:, 1][:, None])[:, 0] * d[:, 0]
                          + sample(gy, a[:, 0][:, None], a[:, 1][:, None])[:, 0] * d[:, 1])
                    val = np.abs(dd).mean()
                    if val > bestv:
                        bestv, best = val, par
    return np.array(best), bestv


def refine(gx, gy, pts2, nrm2, Rp, tp, n, x0, free_h, schedule=(24, 12, 6, 4, 4)):
    x = np.array(x0, float)
    info = {}
    for S in schedule:
        pm, d = project_with_normals(pts2, nrm2, x, Rp, tp, n)
        off, peak, ok = edge_search(gx, gy, pm, d, S)
        if ok.sum() < 100:
            return x, None
        thr = np.percentile(peak[ok], 25)
        ok &= peak >= thr
        q = pm + off[:, None] * d                      # measured edge points
        pq, dq, okq = q[ok], d[ok], ok

        def res(z):
            par = np.array([z[0], z[1], z[2], z[3] if free_h else x[3]])
            a, _ = project_with_normals(pts2[okq], nrm2[okq], par, Rp, tp, n)
            return ((a - pq) * dq).sum(1)

        z0 = x.copy()
        sol = least_squares(res, z0, loss="soft_l1", f_scale=2.0,
                            x_scale=[0.01, 0.002, 0.002, 0.002], max_nfev=60)
        x = sol.x if free_h else np.array([sol.x[0], sol.x[1], sol.x[2], x[3]])
        r = sol.fun
        info = {"n": int(okq.sum()),
                "rms_px": float(np.sqrt((r ** 2).mean())),
                "mad_px": float(np.median(np.abs(r))),
                "inlier_frac_1px": float((np.abs(r) < 1).mean())}
    return x, info


def fit_view(view_dir, T_init, free_h, h_lock, model, quick=False):
    img = cv2.imread(str(view_dir / "color.png"))
    depth = cv2.imread(str(view_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    if img is None or depth is None:
        return None
    img = cv2.undistort(img, K_4K.astype(np.float32), D_4K,
                        newCameraMatrix=K_4K.astype(np.float32))
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.6)
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)

    T = np.asarray(T_init)
    rv, _ = cv2.Rodrigues(T[:3, :3])
    import pose_lib_improved as PL
    c = PL.project(PL.CORNER_3D, rv, T[:3, 3].reshape(3, 1)) / SCALE
    quad = np.array([c[0], c[1], c[3], c[2]])
    pl, _ = table_plane(depth, quad)
    if pl is None:
        return None
    Rp, tp, n = plane_frame(*pl)

    o = Rp.T @ (T[:3, 3] - tp)
    Rz = Rp.T @ T[:3, :3]
    yaw0 = float(np.arctan2(Rz[1, 0], Rz[0, 0]))
    h0 = h_lock if h_lock is not None else float(o[2])
    init = (yaw0, float(o[0]), float(o[1]), h0)

    pts2, nrm2, _ = model
    if quick:
        x0 = np.array(init)
    else:
        x0, _ = coarse(gx, gy, pts2, nrm2, Rp, tp, n, init,
                       yaw_span=18, yaw_step=3, uv_span=0.030, uv_step=0.0075)
    x, info = refine(gx, gy, pts2, nrm2, Rp, tp, n, x0, free_h)
    if info is None:
        return None
    R, t = pose(x, Rp, tp, n)
    Tf = np.eye(4); Tf[:3, :3] = R; Tf[:3, 3] = t
    info.update(yaw_deg=float(np.degrees(x[0]) % 360), u_mm=float(x[1] * 1000),
                v_mm=float(x[2] * 1000), h_mm=float(x[3] * 1000),
                T_cam_board=Tf.tolist(), n_table=[float(q) for q in n])
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-h", action="store_true",
                    help="also solve the board height (4 DoF) instead of locking it")
    ap.add_argument("--h-mm", type=float, default=None,
                    help="lock the board height above the table plane, in mm")
    ap.add_argument("--h-from", default=None,
                    help="calibrate the locked height per pose folder from a "
                         "previous --free-h run, using its mad<1px views")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="skip the coarse search")
    ap.add_argument("--out", default=str(HERE / "out" / "dense_on_table.json"))
    args = ap.parse_args()

    model = board_model.build()
    print(f"model: {int(model[2].max()) + 1} rims, {len(model[0])} points")
    cache = json.load(open(HERE / "out" / "cached_views.json"))
    lab_dir = HERE / "out_annotated"
    lab = {f.stem.split("_to_")[1]: c for c in ("bad", "close", "pixel-perfect")
           for f in (lab_dir / c).glob("match_*.png")}
    data = ROOT / "images" / "images_depth"
    h_lock = None if args.h_mm is None else args.h_mm / 1000.0
    h_by_pose = {}
    if args.h_from:
        prev = json.load(open(args.h_from))
        for p in sorted({r["pose"] for r in prev}):
            g = [r["h_mm"] for r in prev if r["pose"] == p and r["mad_px"] < 1.0]
            if g:
                h_by_pose[p] = float(np.median(g)) / 1000.0
        print("calibrated board height per folder (mm): "
              + "  ".join(f"pose_{p} {v*1000:.2f}" for p, v in h_by_pose.items()))

    rows, k = [], 0
    for p in sorted(cache):
        for view, rec in sorted(cache[p].items()):
            if rec.get("T_cam_board") is None:
                continue
            if args.limit and k >= args.limit:
                break
            k += 1
            r = fit_view(data / f"pose_{p}" / view, rec["T_cam_board"],
                         args.free_h, h_by_pose.get(int(p), h_lock),
                         model, args.quick)
            if r is None:
                print(f"  pose_{p}/{view}  FAILED")
                continue
            r.update(pose=int(p), view=view, label=lab.get(view, "-"))
            rows.append(r)
            print(f"  pose_{p}/{view:<16} {r['label']:<14} "
                  f"rms {r['rms_px']:6.2f} px  mad {r['mad_px']:5.2f}  "
                  f"in1px {r['inlier_frac_1px']:4.2f}  n {r['n']:4d}  "
                  f"yaw {r['yaw_deg']:7.2f}  h {r['h_mm']:6.1f} mm")

    json.dump(rows, open(args.out, "w"), indent=2)
    if not rows:
        return
    print("\n" + "=" * 78)
    rm = np.array([r["mad_px"] for r in rows])
    print(f"{len(rows)} views   rim residual MAD: median {np.median(rm):.2f} px "
          f"({np.median(rm) * 0.209:.3f} mm)   p90 {np.percentile(rm, 90):.2f}")
    for p in sorted({r["pose"] for r in rows}):
        s = [r for r in rows if r["pose"] == p]
        h = np.array([r["h_mm"] for r in s])
        good = [r for r in s if r["mad_px"] < 1.0]
        hg = np.array([r["h_mm"] for r in good])
        print(f"  pose_{p}: n={len(s):2d}  h {np.median(h):6.1f} +- {h.std():5.2f} mm"
              + (f"   |  mad<1px n={len(hg):2d}  h {np.median(hg):6.1f} +- {hg.std():4.2f} mm"
                 if len(hg) > 1 else ""))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
