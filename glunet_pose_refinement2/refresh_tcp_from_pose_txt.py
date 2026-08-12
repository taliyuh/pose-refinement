#!/usr/bin/env python3
"""
refresh_tcp_from_pose_txt.py - re-read T_base_tcp into cached_views.json.

normalize_frames.py rewrites pose.txt, but cached_views.json holds a copy of
T_base_tcp taken before that, so anything reading the cache keeps using the
pre-normalisation frame. This re-reads pose.txt for every cached view and
leaves T_cam_board / T_cam_marker untouched.

Dry run by default; --apply writes, keeping a timestamped backup.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from update_cached_views_all import parse_pose_txt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(HERE / "out" / "cached_views.json"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    cache_p = Path(args.cache)
    cache = json.load(open(cache_p))
    data = ROOT / "images" / "images_depth"

    changed, missing, moved = 0, 0, []
    for p in sorted(cache, key=int):
        for v, rec in cache[p].items():
            f = data / f"pose_{p}" / v / "pose.txt"
            if not f.exists():
                missing += 1
                continue
            T = parse_pose_txt(f)
            old = np.asarray(rec.get("T_base_tcp")) if rec.get("T_base_tcp") else None
            if old is not None:
                d = float(np.linalg.norm(T[:3, 3] - old[:3, 3]) * 1000)
                if d > 0.001:
                    changed += 1
                    moved.append((f"pose_{p}", d))
            rec["T_base_tcp"] = T.tolist()

    by = {}
    for k, d in moved:
        by.setdefault(k, []).append(d)
    for k in sorted(by, key=lambda s: int(s.split("_")[1])):
        print(f"  {k}: {len(by[k])} views moved, median {np.median(by[k]):.1f} mm")
    print(f"{changed} views changed, {missing} missing pose.txt")

    if not args.apply:
        print("\ndry run -- pass --apply to write")
        return
    bak = cache_p.with_suffix(f".{time.strftime('%Y%m%d_%H%M%S')}.bak.json")
    shutil.copy2(cache_p, bak)
    json.dump(cache, open(cache_p, "w"), indent=2)
    print(f"\nbacked up to {bak.name}\nwrote {cache_p}")


if __name__ == "__main__":
    main()
