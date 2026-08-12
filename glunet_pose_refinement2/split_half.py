#!/usr/bin/env python3
"""
split_half.py - internal precision of the dense on-table fit.

Fit the pose twice per view, from two disjoint halves of the hole rims (even
and odd rim index, so both halves span the whole board). The disagreement
between the two independent fits measures the precision of the estimator with
no ground truth, no marker and no robot in the loop -- nothing external can
inflate or flatter it.

Reported in the table plane, where the pose actually lives: dx, dy and dyaw,
with dyaw also expressed as the corner displacement it causes at the board's
203 mm half-diagonal.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "refinement_proposal"))

import board_model  # noqa: E402
import pose_lib_improved as PL  # noqa: E402
from tilt_vs_depth import K_4K, S as SCALE, table_plane  # noqa: E402
from fit_dense_on_table import D_4K, plane_frame, refine  # noqa: E402

HALF_DIAG = 0.2032


def main():
    pts2, nrm2, lid = board_model.build()
    dense = json.load(open(HERE / "out" / "dense_on_table.json"))
    cache = json.load(open(HERE / "out" / "cached_views.json"))
    data = ROOT / "images" / "images_depth"

    rows = []
    for d in dense:
        if d["mad_px"] >= 1.0:
            continue
        vd = data / f"pose_{d['pose']}" / d["view"]
        img = cv2.imread(str(vd / "color.png"))
        depth = cv2.imread(str(vd / "depth.png"), cv2.IMREAD_UNCHANGED)
        if img is None or depth is None:
            continue
        img = cv2.undistort(img, K_4K.astype(np.float32), D_4K,
                            newCameraMatrix=K_4K.astype(np.float32))
        g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.6)
        gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)

        T0 = np.asarray(cache[str(d["pose"])][d["view"]]["T_cam_board"])
        rv, _ = cv2.Rodrigues(T0[:3, :3])
        c = PL.project(PL.CORNER_3D, rv, T0[:3, 3].reshape(3, 1)) / SCALE
        pl, _ = table_plane(depth, np.array([c[0], c[1], c[3], c[2]]))
        if pl is None:
            continue
        Rp, tp, n = plane_frame(*pl)
        x0 = np.array([np.radians(d["yaw_deg"]), d["u_mm"] / 1000,
                       d["v_mm"] / 1000, d["h_mm"] / 1000])

        out = []
        for parity in (0, 1):
            m = (lid % 2) == parity
            x, info = refine(gx, gy, pts2[m], nrm2[m], Rp, tp, n, x0, True,
                             schedule=(6, 4, 4))
            out.append((x, info))
        if any(o[1] is None for o in out):
            continue
        (xa, ia), (xb, ib) = out
        dyaw = np.degrees((xa[0] - xb[0] + np.pi) % (2 * np.pi) - np.pi)
        rows.append({
            "pose": d["pose"], "view": d["view"], "label": d["label"],
            "dx_mm": (xa[1] - xb[1]) * 1000, "dy_mm": (xa[2] - xb[2]) * 1000,
            "dh_mm": (xa[3] - xb[3]) * 1000, "dyaw_deg": dyaw,
            "dyaw_as_mm": abs(np.radians(dyaw)) * HALF_DIAG * 1000,
            "mad": d["mad_px"],
        })
        r = rows[-1]
        print(f"  pose_{r['pose']}/{r['view']:<16}{r['label']:<14} "
              f"dx {r['dx_mm']:+6.3f}  dy {r['dy_mm']:+6.3f}  dh {r['dh_mm']:+6.3f}  "
              f"dyaw {r['dyaw_deg']:+7.4f}deg ({r['dyaw_as_mm']:.3f}mm)")

    if not rows:
        print("nothing to compare")
        return
    json.dump(rows, open(HERE / "out" / "split_half.json", "w"), indent=2)
    print("\n" + "=" * 72)
    print(f"{len(rows)} views, two independent halves of the rim set\n")
    print(f"{'quantity':<26}{'median |diff|':>14}{'p90':>9}{'rms':>9}")
    print("-" * 58)
    for k, nm in [("dx_mm", "in-plane x  (mm)"), ("dy_mm", "in-plane y  (mm)"),
                  ("dyaw_as_mm", "yaw @203mm  (mm)"), ("dh_mm", "height      (mm)"),
                  ("dyaw_deg", "yaw        (deg)")]:
        a = np.abs([r[k] for r in rows])
        print(f"{nm:<26}{np.median(a):>14.3f}{np.percentile(a, 90):>9.3f}"
              f"{np.sqrt((a ** 2).mean()):>9.3f}")
    d = np.array([[r["dx_mm"], r["dy_mm"]] for r in rows])
    print(f"\nin-plane vector disagreement: median {np.median(np.linalg.norm(d, axis=1)):.3f} mm")
    print("single-fit precision is ~1/sqrt(2) of these (each half uses half the data,\n"
          "so the full fit is about sqrt(2) better than either half).")


if __name__ == "__main__":
    main()
