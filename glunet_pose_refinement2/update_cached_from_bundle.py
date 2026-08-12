#!/usr/bin/env python3
"""
update_cached_from_bundle.py - repopulate cached_views.json T_cam_board from a
bundle result.

cached_views.json is read by handeye_board.py and the eval scripts, and it
still carries whatever seeded the views originally. For pose_4..16 that was a
single hardcoded constant pose repeated across all 194 views (the silent
fallback since removed from update_cached_views_all.py), so anything consuming
it inherits a pose that is metres and tens of degrees wrong.

This rewrites T_cam_board from the bundle-adjusted poses, which are the best
estimate available, and leaves T_base_tcp and T_cam_marker untouched -- those
come from pose.txt and from ArUco respectively and are unaffected.

Views the bundle did not solve get T_cam_board = null, deliberately. Leaving a
stale value there is precisely the failure being fixed: downstream code cannot
tell a bad pose from a good one, but it can and does check for null.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(HERE / "out" / "bundle_rims6.json"))
    ap.add_argument("--cache", default=str(HERE / "out" / "cached_views.json"))
    ap.add_argument("--apply", action="store_true",
                    help="write the file; default is a dry run")
    args = ap.parse_args()

    bundle_p, cache_p = Path(args.bundle), Path(args.cache)
    if not bundle_p.exists():
        sys.exit(f"no bundle at {bundle_p}")
    if not cache_p.exists():
        sys.exit(f"no cache at {cache_p}")

    B = json.load(open(bundle_p))
    views = B["views"] if isinstance(B, dict) else B
    fitted = {(str(r["pose"]), r["view"]): r for r in views}
    cache = json.load(open(cache_p))

    n_set = n_null = n_kept_marker = 0
    per_pose = {}
    for p in sorted(cache, key=int):
        for v in cache[p]:
            rec = cache[p][v]
            r = fitted.get((p, v))
            if r is not None:
                rec["T_cam_board"] = r["T_cam_board"]
                rec["board_src"] = "bundle"
                rec["board_mad_px"] = r.get("mad_px")
                n_set += 1
                per_pose.setdefault(p, [0, 0])[0] += 1
            else:
                rec["T_cam_board"] = None
                rec["board_src"] = None
                rec.pop("board_mad_px", None)
                n_null += 1
                per_pose.setdefault(p, [0, 0])[1] += 1
            if rec.get("T_cam_marker") is not None:
                n_kept_marker += 1

    print(f"{'pose':8} {'from bundle':>12} {'nulled':>8}")
    for p in sorted(per_pose, key=int):
        a, b = per_pose[p]
        print(f"pose_{p:<3} {a:12d} {b:8d}")
    print(f"\n{n_set} views set from bundle, {n_null} nulled, "
          f"{n_kept_marker} marker poses preserved")

    if not args.apply:
        print("\ndry run -- pass --apply to write")
        return
    bak = cache_p.with_suffix(f".{time.strftime('%Y%m%d_%H%M%S')}.bak.json")
    shutil.copy2(cache_p, bak)
    json.dump(cache, open(cache_p, "w"), indent=2)
    print(f"\nbacked up to {bak.name}\nwrote {cache_p}")


if __name__ == "__main__":
    main()
