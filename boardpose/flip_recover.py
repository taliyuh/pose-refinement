#!/usr/bin/env python3
"""
flip_recover.py - find and repair 180-degree yaw flips in the fitted poses.

The board is nearly 2-fold symmetric about its centre, so the chamfer search has
a second, almost-as-good basin at yaw + 180 deg. A handful of views land in it.
The signature is unambiguous: every board corner moves by

    2 * sqrt(160^2 + 125^2) = 406 mm

so a flipped view sits ~410-440 mm from its folder's consensus while every good
view sits within a few mm. Measured across the v2 bundle, the flipped views are
also ranked 1, 2, 3 and 8 of 237 by rim residual -- the wrong basin fits
measurably WORSE (mad 1.45-1.74 px against a 0.68 px median).

That last fact is why these are repaired instead of discarded. If the board were
exactly symmetric the two basins would score identically and the flip would be
unrecoverable; because it is not, the correct pose exists and is better. Seeding
the refiner with the flipped pose and keeping whichever of the two wins on rim
residual therefore recovers the view rather than throwing it away.

Detection is deliberately geometric (deviation from folder consensus) rather
than a residual threshold: a residual gate alone cannot tell a flip from a
merely blurry view, and the two need different treatment.
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
import board_scatter as BS  # noqa: E402
from fit_rims6 import RT, refine  # noqa: E402
from camera import K_4K, D_4K  # noqa: E402

FLIP_MM = 50.0     # deviation from folder consensus that counts as gross


def flip_pose(T):
    """T rotated 180 deg about the board's own z axis."""
    F = np.diag([-1.0, -1.0, 1.0, 1.0])
    return T @ F


def detect(src, cache_p, gate=1.2, thresh=FLIP_MM):
    """Views whose board pose is grossly inconsistent with their folder."""
    groups, _ = BS.board_poses(src, cache_p, gate=gate)
    out = []
    for p in sorted(groups):
        items = groups[p]
        Ts = [t for _, t, _, _ in items]
        mads = np.array([m if m is not None else 9.0 for _, _, m, _ in items])
        ok = mads <= gate
        C = BS.consensus([T for T, k in zip(Ts, ok) if k]
                         if ok.sum() >= 3 else Ts)
        for (v, T, m, Tc), d in zip(items,
                                    [BS.corner_dev(T, C) for T in Ts]):
            if d >= thresh:
                ang, az = BS.yaw_about_normal(T, C)
                out.append({"pose": p, "view": v, "dev_mm": float(d),
                            "rot_deg": float(ang), "axis_z": float(az),
                            "mad_px": m})
    return out


def refit(pose, view, T_cam_board, data):
    """Re-refine from the flipped seed; return the better of the two."""
    vd = data / f"pose_{pose}" / view
    img = cv2.imread(str(vd / "color.png"))
    if img is None:
        return None
    img = cv2.undistort(img, K_4K.astype(np.float32), D_4K,
                        newCameraMatrix=K_4K.astype(np.float32))
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.6)
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    P3, N2, lvl, loop = board_model.build_levels()

    best = None
    for tag, T in (("as-is", T_cam_board), ("flipped", flip_pose(T_cam_board))):
        rv, _ = cv2.Rodrigues(T[:3, :3])
        x, info = refine(gx, gy, P3, N2, lvl, loop,
                         np.concatenate([rv.ravel(), T[:3, 3]]))
        if info is None:
            continue
        if best is None or info["mad_px"] < best[1]["mad_px"]:
            R, t = RT(x)
            M = np.eye(4); M[:3, :3], M[:3, 3] = R, t
            best = (tag, info, M)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "out" / "bundle_rims6_v2.json"))
    ap.add_argument("--cache", default=str(HERE / "out" / "cached_views.json"))
    ap.add_argument("--out", default=None,
                    help="write a corrected copy of --src here")
    ap.add_argument("--detect-only", action="store_true")
    args = ap.parse_args()

    bad = detect(args.src, args.cache)
    print(f"{len(bad)} views deviate >= {FLIP_MM:.0f} mm from their folder:\n")
    for b in bad:
        kind = ("yaw flip" if b["axis_z"] > 0.9 and b["rot_deg"] > 90
                else "gross error")
        print(f"  pose_{b['pose']:<3} {b['view']:16} dev {b['dev_mm']:7.1f} mm  "
              f"rot {b['rot_deg']:6.2f} deg  mad {b['mad_px']:.2f} px   {kind}")
    if args.detect_only or not bad:
        return

    B = json.load(open(args.src))
    views = B["views"] if isinstance(B, dict) else B
    idx = {(r["pose"], r["view"]): r for r in views}
    data = ROOT / "images" / "images_depth"

    print("\nre-refining from the flipped seed:")
    n_fix = 0
    for b in bad:
        r = idx.get((b["pose"], b["view"]))
        if r is None:
            continue
        res = refit(b["pose"], b["view"], np.asarray(r["T_cam_board"]), data)
        if res is None:
            print(f"  pose_{b['pose']}/{b['view']}  could not refine")
            continue
        tag, info, M = res
        print(f"  pose_{b['pose']:<3} {b['view']:16} {tag:8} "
              f"mad {b['mad_px']:.2f} -> {info['mad_px']:.2f} px  n {info['n']}")
        if tag == "flipped" and info["mad_px"] < b["mad_px"]:
            r["T_cam_board"] = M.tolist()
            r["mad_px"] = info["mad_px"]
            r["rms_px"] = info["rms_px"]
            r["inlier_frac_1px"] = info["inlier_frac_1px"]
            r["n"] = info["n"]
            r["flip_corrected"] = True
            n_fix += 1

    print(f"\n{n_fix}/{len(bad)} views corrected")
    if args.out:
        json.dump(B, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
