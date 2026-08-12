#!/usr/bin/env python3
"""
eval_consistency.py - ground-truth-free accuracy check.

Within a pose_* folder the board and the ArUco marker are both static, so

    T_marker_board = inv(T_cam_marker) @ T_cam_board

must be IDENTICAL for every view. Its scatter across views is a direct measure
of board-pose error that needs no robot, no hand-eye and no touch probe -- and
so is not limited by the ~1.5-2.5 mm accuracy of the probed ground truth.

(The marker pose carries its own error, so this bounds the combined error of
the two; it cannot flatter the board estimate.)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def rot_mean(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def ang(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def scatter(Ts):
    Ts = np.asarray(Ts)
    if len(Ts) < 2:
        return None
    Rm = rot_mean(Ts[:, :3, :3])
    tm = np.median(Ts[:, :3, 3], axis=0)
    dt = np.linalg.norm(Ts[:, :3, 3] - tm, axis=1) * 1000
    da = np.array([ang(Rm.T @ T[:3, :3]) for T in Ts])
    return {"n": len(Ts), "t_med": float(np.median(dt)), "t_p90": float(np.percentile(dt, 90)),
            "t_rms": float(np.sqrt((dt ** 2).mean())),
            "r_med": float(np.median(da)), "r_p90": float(np.percentile(da, 90))}


def main():
    cache = json.load(open(HERE / "out" / "cached_views.json"))
    dense = {(r["pose"], r["view"]): r
             for r in json.load(open(HERE / "out" / "dense_on_table.json"))}

    sets = {"4-corner PnP (current pipeline)": {}, "dense on-table fit": {},
            "dense on-table, mad < 1.0 px": {}}
    for p in sorted(cache):
        for view, rec in cache[p].items():
            M = rec.get("T_cam_marker")
            if M is None:
                continue
            Mi = np.linalg.inv(np.asarray(M))
            if rec.get("T_cam_board") is not None:
                sets["4-corner PnP (current pipeline)"].setdefault(int(p), []).append(
                    Mi @ np.asarray(rec["T_cam_board"]))
            d = dense.get((int(p), view))
            if d is not None:
                T = Mi @ np.asarray(d["T_cam_board"])
                sets["dense on-table fit"].setdefault(int(p), []).append(T)
                if d["mad_px"] < 1.0:
                    sets["dense on-table, mad < 1.0 px"].setdefault(int(p), []).append(T)

    print("scatter of T_marker_board across views of the same static scene")
    print("(median / p90 deviation from the per-folder robust mean)\n")
    hdr = f"{'method':<34}{'pose':>6}{'n':>4}{'trans med':>11}{'p90':>8}{'rot med':>10}{'p90':>8}"
    print(hdr); print("-" * len(hdr))
    for name, byp in sets.items():
        for p in sorted(byp):
            s = scatter(byp[p])
            if s:
                print(f"{name:<34}{p:>6}{s['n']:>4}{s['t_med']:>9.2f}mm{s['t_p90']:>7.2f}"
                      f"{s['r_med']:>9.3f}°{s['r_p90']:>7.3f}")
        print()


if __name__ == "__main__":
    main()
