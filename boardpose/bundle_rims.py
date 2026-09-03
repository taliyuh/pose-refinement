#!/usr/bin/env python3
"""
bundle_rims.py - multi-view bundle adjustment over the hole-rim observations.

What is actually shared between views: the board is static within a folder, but
its geometry is already known exactly from the STL, so each view's pose is
independently determined. The genuinely shared unknown is the CAMERA ITSELF.

So this solves, jointly over every view of every folder:

    shared      fx, fy, cx, cy, k1, k2, p1, p2, k3        (9)
    shared      rim bias, per contour level                (2-6)
    per view    full 6-DoF T_cam_board                    (6 x N)

against ~50k subpixel rim-edge observations. That is Zhang-style calibration
carried out on the actual measurement object rather than a separate
checkerboard, which is what removes the systematic bias that no amount of
per-view fitting can reach. The current K_4K is exactly 3x the 720p matrix,
i.e. it was rescaled rather than recalibrated -- this checks it.

Two other things change versus the old on-table fitter:

  * tilt is FREE here. The table constraint was needed because 4 corners cannot
    see tilt; 1184 subpixel points over a 320 x 250 mm board at 37 deg angular
    extent can. Depth is then demoted from an input to an independent check.
  * distortion is estimated, so points are projected INTO the raw image and the
    edge search runs on unresampled pixels.
  * the RIM BIAS is estimated. The STL outline is not where the intensity edge
    is -- the top hole edge is chamfered and the pocket floor rim sits at the
    base of a 22 mm wall -- and because tilt is triangulated from the 22 mm
    baseline between the two levels, the differential bias between them is a
    tilt error by construction. It was worth 0.68 deg of tilt bias and 0.18 deg
    of tilt scatter, the latter being the entire per-view tilt error seen in the
    robot frame. See board_model.rim_bias(). It is identifiable here rather than
    degenerate with focal length because a radial offset changes each rim's SIZE
    without changing the SPACING between rims, which a scale change cannot do.

The marker is deliberately kept out of the bundle: folding in T_cam_marker
would contaminate the board poses with ArUco error. It is used afterwards, as
an independent check, exactly as before.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import board_model  # noqa: E402
from camera import K_4K, D_4K, edge_search  # noqa: E402

NINTR = 9
NBIAS = board_model.NBIAS
MARGIN = 140


# ------------------------------------------------------------------ project

def project(P3, rvec, tvec, intr):
    """Board points -> pixels, with the full 5-term Brown-Conrady distortion."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = intr
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    in_camera = P3 @ R.T + tvec

    depth = np.maximum(in_camera[:, 2], 1e-6)
    x, y = in_camera[:, 0] / depth, in_camera[:, 1] / depth
    r2 = x * x + y * y

    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    x_distorted = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([fx * x_distorted + cx, fy * y_distorted + cy], 1)


def project_with_normals(pts3, nrm2, rvec, tvec, intr):
    """Projected points, plus the image-space direction each normal maps to.

    The normal is carried through the distortion numerically -- step 1 mm along
    it in board space and difference -- because an analytic push-forward would
    have to differentiate the distortion polynomial.
    """
    step = np.concatenate([nrm2 * 1e-3, np.zeros((len(nrm2), 1))], 1)
    at_point = project(pts3, rvec, tvec, intr)
    at_step = project(pts3 + step, rvec, tvec, intr)

    direction = at_step - at_point
    length = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
    return at_point, direction / length


# ------------------------------------------------------------------ views

class View:
    __slots__ = ("pose", "name", "label", "gx", "gy", "ox", "oy", "rvec", "tvec")


def _well_fitted_rows(rows, max_mad):
    """Per-view fits whose rim residual is good enough to bundle."""
    kept = []
    for row in rows:
        if row["mad_px"] < max_mad:
            kept.append(row)
    return kept


def crop_bounds(projected, image_shape, margin=MARGIN):
    """Bounding box of the projected board, padded and clipped to the image."""
    x0 = int(max(0, projected[:, 0].min() - margin))
    x1 = int(min(image_shape[1], projected[:, 0].max() + margin))
    y0 = int(max(0, projected[:, 1].min() - margin))
    y1 = int(min(image_shape[0], projected[:, 1].max() + margin))
    return x0, x1, y0, y1


def gradient_images(image, x0, x1, y0, y1):
    """Scharr gradients of the cropped, lightly blurred board region."""
    blurred = cv2.GaussianBlur(image[y0:y1, x0:x1], (0, 0), 1.6)
    return (cv2.Scharr(blurred, cv2.CV_32F, 1, 0),
            cv2.Scharr(blurred, cv2.CV_32F, 0, 1))


def _factory_intrinsics():
    """K_4K and D_4K flattened into the 9-vector `project` expects."""
    return (list(np.diag(K_4K)[:2]) + [K_4K[0, 2], K_4K[1, 2]] + list(D_4K))


def load_views(init_json, max_mad, data):
    fitted = _well_fitted_rows(json.load(open(init_json)), max_mad)
    pts3 = board_model.build_levels()[0]
    seed_intrinsics = _factory_intrinsics()

    views = []
    for row in fitted:
        view_dir = data / f"pose_{row['pose']}" / row["view"]
        image = cv2.imread(str(view_dir / "color.png"), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        T_cam_board = np.asarray(row["T_cam_board"])
        view = View()
        view.pose, view.name, view.label = row["pose"], row["view"], row["label"]
        view.rvec = cv2.Rodrigues(T_cam_board[:3, :3])[0].ravel()
        view.tvec = T_cam_board[:3, 3].copy()

        # Crop to the board plus margin so 67 gradient images fit in RAM.
        projected = project(pts3, view.rvec, view.tvec, seed_intrinsics)
        x0, x1, y0, y1 = crop_bounds(projected, image.shape)
        if x1 - x0 < 200 or y1 - y0 < 200:
            continue

        view.gx, view.gy = gradient_images(image, x0, x1, y0, y1)
        view.ox, view.oy = x0, y0
        views.append(view)
    return views


# ------------------------------------------------------------------ bundle

def measure(views, intr, pts3, nrm2, S, level=None, loop=None, bias=None):
    """One ICP-style association pass: detect the edge each rim point sits on."""
    observations = []
    for view in views:
        R = cv2.Rodrigues(view.rvec)[0]
        model_points = (pts3 if bias is None or level is None else
                        board_model.biased_pts(pts3, nrm2, level, R,
                                               view.tvec, bias))
        projected, direction = project_with_normals(model_points, nrm2,
                                                    view.rvec, view.tvec, intr)
        in_crop = projected - [view.ox, view.oy]
        offset, peak, is_usable = edge_search(view.gx, view.gy, in_crop,
                                              direction, S)

        if level is not None:
            is_usable &= board_model.visible(pts3, nrm2, level, loop, R,
                                             view.tvec)
        is_usable &= _inside_search_window(in_crop, view.gx.shape, S)

        # Drop the weakest quarter of the surviving edges: a rim point on a
        # faint gradient is a worse observation than no observation at all.
        if is_usable.sum() > 50:
            is_usable &= peak >= np.percentile(peak[is_usable], 25)

        observations.append({
            "idx": np.nonzero(is_usable)[0],
            "q": (projected + offset[:, None] * direction)[is_usable],
            "d": direction[is_usable]})
    return observations


def _inside_search_window(in_crop, image_shape, S):
    """Points close enough to the crop edge that the search would run off it."""
    height, width = image_shape
    return ((in_crop[:, 0] > S + 2) & (in_crop[:, 0] < width - S - 2)
            & (in_crop[:, 1] > S + 2) & (in_crop[:, 1] < height - S - 2))


NSH = NINTR + NBIAS          # shared parameters, laid out first


def pack(views, intr, bias):
    """Shared parameters first, then each view's 6-DoF pose."""
    return np.concatenate([intr, bias]
                          + [np.concatenate([v.rvec, v.tvec]) for v in views])


def unpack(x, views):
    """Write the per-view poses back onto `views`; return (intrinsics, bias)."""
    for i, view in enumerate(views):
        start = _view_offset(i)
        view.rvec, view.tvec = x[start:start + 3], x[start + 3:start + 6]
    return x[:NINTR], x[NINTR:NSH]


def _view_offset(view_index):
    return NSH + 6 * view_index


def residuals(x, views, obs, pts3, nrm2, level, fit_intr, intr0, free_bias, bias0):
    intr = x[:NINTR] if fit_intr else intr0
    bias = _current_bias(x, free_bias, bias0)

    blocks = []
    for i, (view, obs_i) in enumerate(zip(views, obs)):
        if len(obs_i["idx"]) == 0:
            continue
        start = _view_offset(i)
        rvec, tvec = x[start:start + 3], x[start + 3:start + 6]

        idx = obs_i["idx"]
        model_points = pts3[idx]
        if level is not None:
            model_points = board_model.biased_pts(
                model_points, nrm2[idx], level[idx],
                cv2.Rodrigues(rvec)[0], tvec, bias)

        projected, _ = project_with_normals(model_points, nrm2[idx],
                                            rvec, tvec, intr)
        blocks.append(((projected - obs_i["q"]) * obs_i["d"]).sum(1))
    return np.concatenate(blocks) if blocks else np.zeros(1)


def _current_bias(x, free_bias, bias0):
    """Only the free components come from x; the rest stay at their seed, so a
    zero-Jacobian column can never drift."""
    bias = np.asarray(bias0, float).copy()
    for component in free_bias:
        bias[component] = x[NINTR + component]
    return bias


def sparsity(views, obs, fit_intr, free_bias):
    row_count = sum(len(obs_i["idx"]) for obs_i in obs)
    pattern = lil_matrix((row_count, NSH + 6 * len(views)), dtype=np.uint8)

    row = 0
    for i, obs_i in enumerate(obs):
        rows_here = len(obs_i["idx"])
        if rows_here == 0:
            continue
        if fit_intr:
            pattern[row:row + rows_here, :NINTR] = 1
        for component in free_bias:
            pattern[row:row + rows_here, NINTR + component] = 1
        start = _view_offset(i)
        pattern[row:row + rows_here, start:start + 6] = 1
        row += rows_here
    return pattern


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(HERE / "out" / "dense_on_table.json"))
    ap.add_argument("--max-mad", type=float, default=1.5,
                    help="only seed from views the on-table fit already explains")
    ap.add_argument("--fix-intrinsics", action="store_true")
    ap.add_argument("--rim-bias", default="b0",
                    choices=["none", "b0", "b0sym", "b0b1"],
                    help="none: the pre-2026-08 model. "
                         "b0: constant offset per contour level (default). "
                         "b0sym: b0 plus the SYMMETRIC view terms b2|s| and "
                         "b3|u_xy| -- safe but USELESS, they improve the pixel "
                         "fit and make the pose worse. "
                         "b0b1: b0 plus the signed term -- REJECTED, b1*s "
                         "translates the rim. See board_model.rim_bias().")
    ap.add_argument("--top-only", action="store_true",
                    help="use only the top rims (single plane) for comparison")
    ap.add_argument("--out", default=str(HERE / "out" / "bundle_rims.json"))
    args = ap.parse_args()

    if args.top_only:
        p2, nrm2, loop = board_model.build()
        pts3 = board_model.pts3(p2); level = np.zeros(len(p2), int)
        # One level means no differential, so the rim bias has nothing to
        # triangulate against and b0 collapses into the standoff. This mode
        # exists only as the 1-plane comparison; keep it as it was.
        if args.rim_bias != "none":
            print("--top-only: rim bias is not identifiable from one level, "
                  "disabling it")
            args.rim_bias = "none"
    else:
        pts3, nrm2, level, loop = board_model.build_levels()
    data = ROOT / "images" / "images_depth"
    t0 = time.time()
    views = load_views(args.init, args.max_mad, data)
    print(f"{len(views)} views, {len(pts3)} rim points each "
          f"({time.time() - t0:.0f}s to load)")

    intr0 = np.array([K_4K[0, 0], K_4K[1, 1], K_4K[0, 2], K_4K[1, 2],
                      *D_4K.astype(np.float64)])
    intr = intr0.copy()
    fit_intr = not args.fix_intrinsics
    bias0 = board_model.RIM_BIAS0.copy()
    if args.rim_bias == "none":
        bias0 = np.zeros(NBIAS)
    bias = bias0.copy()
    free = {"none": [], "b0": [0, 1], "b0sym": [0, 1, 4, 5, 6, 7],
            "b0b1": [0, 1, 2, 3]}[args.rim_bias]

    for it, S in enumerate((10, 6, 4, 4)):
        obs = measure(views, intr, pts3, nrm2, S, level, loop, bias)
        x0 = pack(views, intr, bias)
        sp = sparsity(views, obs, fit_intr, free)
        sol = least_squares(residuals, x0, jac_sparsity=sp, loss="soft_l1",
                            f_scale=1.5, method="trf", max_nfev=40, verbose=0,
                            args=(views, obs, pts3, nrm2, level, fit_intr,
                                  intr0, free, bias0))
        fi, fb = unpack(sol.x, views)
        intr = fi if fit_intr else intr0
        bias = bias0.copy()
        for b in free:
            bias[b] = fb[b]
        r = sol.fun
        print(f"  iter {it} (S={S:2d})  n={len(r):6d}  "
              f"rms {np.sqrt((r ** 2).mean()):.3f} px  "
              f"mad {np.median(np.abs(r)):.3f} px  "
              f"in1px {(np.abs(r) < 1).mean():.2f}   [{time.time() - t0:.0f}s]")

    print("\nintrinsics" + ("  (refined)" if fit_intr else "  (held fixed)"))
    names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
    for nm, a, b in zip(names, intr0, intr):
        print(f"  {nm:<3} {a:>14.6f} -> {b:>14.6f}   delta {b - a:+.6f}")

    print(f"\nrim bias  (--rim-bias {args.rim_bias})"
          + "   [mm along each rim's outward normal]")
    for i, (nm, a, b) in enumerate(zip(board_model.RIM_BIAS_NAMES, bias0, bias)):
        tag = "" if i in free else "   (held)"
        print(f"  {nm:<9} {a * 1000:>9.4f} -> {b * 1000:>9.4f} mm{tag}")
    dif = (bias[1] - bias[0]) * 1000
    print(f"  differential b0 (floor - top): {dif:+.4f} mm across the 22 mm "
          f"level baseline = {np.degrees(np.arctan2(abs(dif), 22.0)):.4f} deg "
          f"of tilt it was costing")

    rows = []
    obs = measure(views, intr, pts3, nrm2, 4, level, loop, bias)
    for v, o in zip(views, obs):
        if len(o["idx"]) == 0:
            continue
        P = board_model.biased_pts(pts3[o["idx"]], nrm2[o["idx"]], level[o["idx"]],
                                   cv2.Rodrigues(v.rvec)[0], v.tvec, bias)
        a, _ = project_with_normals(P, nrm2[o["idx"]], v.rvec, v.tvec, intr)
        r = ((a - o["q"]) * o["d"]).sum(1)
        T = np.eye(4); T[:3, :3] = cv2.Rodrigues(v.rvec)[0]; T[:3, 3] = v.tvec
        rows.append({"pose": v.pose, "view": v.name, "label": v.label,
                     "n": int(len(r)), "rms_px": float(np.sqrt((r ** 2).mean())),
                     "mad_px": float(np.median(np.abs(r))),
                     "T_cam_board": T.tolist()})
    json.dump({"intrinsics": list(map(float, intr)),
               "intrinsics_names": names,
               "rim_bias": list(map(float, bias)),
               "rim_bias_names": board_model.RIM_BIAS_NAMES,
               "views": rows},
              open(args.out, "w"), indent=2)
    m = np.array([r["mad_px"] for r in rows])
    print(f"\n{len(rows)} views   rim residual MAD: median {np.median(m):.3f} px "
          f"({np.median(m) * 0.209:.3f} mm)   p90 {np.percentile(m, 90):.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
