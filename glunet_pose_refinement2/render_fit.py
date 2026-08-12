#!/usr/bin/env python3
"""
render_fit.py - visual check of the fitted board pose.

For each view writes one panel:
  * left  : full frame, everything drawn under the FITTED pose.
            green thin = board outer outline (320 x 250 mm)
            green dots = top rims (z = 25 mm)
            cyan  dots = pocket floor rims (z = 3 mm), only where visible
  * right : native-resolution zooms on four pockets, so the subpixel fit can
            actually be judged. Each rim point is drawn 1 px wide.

The outline and the rims come from the same pose, so the panel answers two
different questions at once: the rims show whether the fit is subpixel, and the
outline shows whether the board as a whole is tracked -- a fit can be wrong by
a whole pocket pitch with every rim still sitting on *a* rim.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))


import board_model  # noqa: E402
from bundle_rims import project, project_with_normals  # noqa: E402
from tilt_vs_depth import K_4K  # noqa: E402
from fit_dense_on_table import D_4K  # noqa: E402

ZOOM = 3
TILE = 230


def draw(img, pts, colour, r=2):
    ok = (pts[:, 0] > 0) & (pts[:, 0] < img.shape[1]) & \
         (pts[:, 1] > 0) & (pts[:, 1] < img.shape[0])
    for x, y in pts[ok].astype(int):
        cv2.circle(img, (x, y), r, colour, -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "out" / "bundle_2level.json"))
    ap.add_argument("--outdir", default=str(HERE / "out_overlays"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    B = json.load(open(args.src))
    if isinstance(B, list):
        # flat per-view list (fit_rims6.py / fit_dense_on_table.py): no bundle
        # has run yet, so the camera is the factory one.
        B = {"views": B,
             "intrinsics": [K_4K[0, 0], K_4K[1, 1], K_4K[0, 2], K_4K[1, 2], *D_4K]}
    intr = np.array(B["intrinsics"])
    pts3, nrm2, level, loop = board_model.build_levels()
    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)

    for i, r in enumerate(B["views"]):
        if args.limit and i >= args.limit:
            break
        vd = ROOT / "images" / "images_depth" / f"pose_{r['pose']}" / r["view"]
        img = cv2.imread(str(vd / "color.png"))
        if img is None:
            continue
        T = np.asarray(r["T_cam_board"])
        rv = cv2.Rodrigues(T[:3, :3])[0].ravel()
        tv = T[:3, 3]
        vis = board_model.visible(pts3, nrm2, level, loop, T[:3, :3], tv)
        P = project(pts3, rv, tv, intr)

        # Board outline under the FITTED pose, thin. This is the whole-board
        # check: the rims can all sit correctly while the board as a whole is
        # misplaced, and vice versa, so the outline is drawn from the same pose
        # the rims are drawn from. It replaces the old 4-corner PnP outline,
        # which was drawn from the cached seed -- for pose_4..16 that seed was
        # the constant fallback, so the rectangle was simply in the wrong place
        # and told you nothing about the fit.
        oc = project(board_model.outline(), rv, tv, intr)
        cv2.polylines(img, [oc.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)

        # zoom tiles before the contours are drawn thick
        cen = np.array([pts3[(level == 0) & (loop == g)][:, :2].mean(0)
                        for g in range(20)])
        cen3 = np.concatenate([cen, np.full((20, 1), 0.025)], 1)
        pc = project(cen3, rv, tv, intr)
        pick = [0, 5, 12, 19]
        tiles = []
        for k in pick:
            cx, cy = pc[k]
            h = TILE // (2 * ZOOM)
            x0, y0 = int(cx) - h, int(cy) - h
            x1, y1 = x0 + 2 * h, y0 + 2 * h
            if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
                tiles.append(np.zeros((TILE, TILE, 3), np.uint8)); continue
            t = cv2.resize(img[y0:y1, x0:x1], (TILE, TILE), interpolation=cv2.INTER_NEAREST)
            for m, col in ((level == 0, (0, 255, 0)), ((level == 1) & vis, (255, 255, 0))):
                q = (P[m] - [x0, y0]) * (TILE / (2 * h))
                draw(t, q, col, 1)
            cv2.rectangle(t, (0, 0), (TILE - 1, TILE - 1), (60, 60, 60), 1)
            tiles.append(t)

        draw(img, P[level == 0], (0, 255, 0), 3)
        draw(img, P[(level == 1) & vis], (255, 255, 0), 3)
        left = cv2.resize(img, (1280, 720))
        cv2.putText(left, f"pose_{r['pose']}/{r['view']}  {r['label']}", (12, 30),
                    0, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(left, f"rim residual MAD {r['mad_px']:.2f} px "
                          f"({r['mad_px'] * 0.209:.3f} mm)   n={r['n']}", (12, 60),
                    0, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(left, "green top rims + board outline   cyan pocket floors"
                          "   (all from the fitted pose)",
                    (12, 700), 0, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

        right = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        right = cv2.resize(right, (720, 720), interpolation=cv2.INTER_NEAREST)
        panel = np.hstack([left, right])
        cv2.imwrite(str(outdir / f"fit_pose{r['pose']}_{r['view']}.png"), panel)
    print(f"wrote {len(list(outdir.glob('fit_*.png')))} panels to {outdir}")


if __name__ == "__main__":
    main()
