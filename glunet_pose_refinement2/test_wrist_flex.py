#!/usr/bin/env python3
"""
test_wrist_flex.py - does a configuration-dependent hand-eye actually work?

After the rim-bias fix, theta_6 became the strongest predictor of what is left
in the robot-frame residual (r = +0.337, against a max of 0.15 for every other
joint angle, every effort, reach, height and capture order). theta_6 rolls the
camera about the tool axis, swinging a stick-mounted camera's mass around, so a
hand-eye that varies with it is physically the right shape.

It is also a close cousin of the `flex` model deleted from handeye_board.py, so
it gets tested the way that one should have been:

    THE GATE. In-sample residual is not evidence -- 12 free parameters will
    always lower it. The model ships only if leave-one-folder-out improves,
    i.e. if parameters fitted on 10 folders transfer UNCHANGED to the 11th.
    That is what separates a property of the arm from a parameter absorbing
    per-folder error. Joint offsets pass it 11/11.

Both candidate regressors are tested, because "theta_6 is special" and "gravity
moved" are different claims and the data can tell them apart.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import kinematic_calib as KC  # noqa: E402


def lofo(views, flex, poses, prior):
    """Leave-one-folder-out: fit on 10 folders, apply UNCHANGED to the 11th.

    Each model transfers its OWN parameters. That matters: dtheta fitted
    alongside a runaway flex term is itself contaminated, so using it as the
    baseline's offsets would compare the flex model against a sabotaged
    reference rather than against the model it has to beat.
    """
    base, corr = [], []
    for p in poses:
        tr = [v for v in views if v[0] != p]
        te = [v for v in views if v[0] == p]
        if len(te) < 4:
            continue
        t_off = KC.solve(tr, use_offset=True)                 # offsets alone
        t_flex = KC.solve(tr, use_offset=True, flex=flex, flex_prior=prior)
        held_b = KC.solve(te, use_offset=False,
                          fixed_dtheta=t_off["dtheta"])
        held_c = KC.solve(te, use_offset=False,
                          fixed_dtheta=t_flex["dtheta"],
                          flex=flex, fixed_flex=t_flex["flex"])
        b = np.median(held_b["per_view_mm"])
        c = np.median(held_c["per_view_mm"])
        base.append(b); corr.append(c)
        print(f"    pose_{p:<3} n={len(te):3d}   offsets only {b:6.3f} -> "
              f"+flex {c:6.3f} mm   {'better' if c < b else 'WORSE'}")
    return np.array(base), np.array(corr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "out" / "bundle_rims6_v3.json"))
    ap.add_argument("--poses", default="4,6,7,9,10,11,12,13,14,15,16")
    ap.add_argument("--flex-prior", type=float, default=1.0,
                    help="prior on the deflection amplitude [mm at the board]. "
                         "A 135 mm stick cannot deflect more than a fraction "
                         "of a mm; 1.0 is already generous. Set 0 to fit it "
                         "unconstrained and watch it run away.")
    ap.add_argument("--out", default=str(HERE / "out" / "wrist_flex_test.json"))
    args = ap.parse_args()
    poses = [int(s) for s in args.poses.split(",")]

    views = KC.load(args.src, poses)
    base0 = KC.solve(views, use_offset=False)
    e0 = base0["per_view_mm"]
    keep = e0 < max(5 * np.median(e0), 5.0)
    views = [v for v, k in zip(views, keep) if k]
    print(f"{len(views)} views, {int((~keep).sum())} gross outliers rejected\n")

    result = {}
    print("IN-SAMPLE (not evidence -- more parameters always win here)")
    print(f"  {'model':26} {'params':>7} {'median':>8} {'p90':>8}")
    prior = (args.flex_prior / 1000.0) or None
    print(f"  (deflection amplitude prior: "
          + (f"{args.flex_prior:.2f} mm at the board)" if prior else "NONE)"))
    for name, flex, npar in [("joint offsets only", None, 4),
                             ("+ wrist flex (theta_6)", "wrist", 4 + 12),
                             ("+ gravity flex (TCP)", "gravity", 4 + 18),
                             ("+ PLACEBO fake_x", "fake_x", 4 + 18),
                             ("+ PLACEBO fake_y", "fake_y", 4 + 18)]:
        info = KC.solve(views, use_offset=True, flex=flex, flex_prior=prior)
        e = info["per_view_mm"]
        print(f"  {name:26} {npar:7d} {np.median(e):8.3f} "
              f"{np.percentile(e, 90):8.3f} mm")
        result[name] = {"params": npar, "median_mm": float(np.median(e)),
                        "p90_mm": float(np.percentile(e, 90))}
        if flex:
            M = info["flex"]
            rot = np.degrees(np.linalg.norm(M[:3], axis=0))
            tr = np.linalg.norm(M[3:], axis=0) * 1000
            board_mm = max(rot.max() * np.pi / 180 * 470, tr.max())
            flag = "" if board_mm < 2.0 else "   <-- implausible for a 135 mm stick"
            print(f"    amplitude: rotation up to {rot.max():.4f} deg, "
                  f"translation up to {tr.max():.4f} mm"
                  f"  = {board_mm:.3f} mm at the board{flag}")
            result[name]["rot_amplitude_deg"] = float(rot.max())
            result[name]["trans_amplitude_mm"] = float(tr.max())

    print("\nTHE GATE: leave-one-folder-out, parameters transferred unchanged")
    for name, flex in [("wrist flex (theta_6)", "wrist"),
                       ("gravity flex (TCP)", "gravity"),
                       ("PLACEBO fake_x (no gravity meaning)", "fake_x"),
                       ("PLACEBO fake_y (no gravity meaning)", "fake_y")]:
        print(f"\n  {name}")
        b, c = lofo(views, flex, poses, prior)
        imp = int((c < b).sum())
        print(f"    overall: {np.median(b):.3f} -> {np.median(c):.3f} mm median, "
              f"{imp}/{len(b)} folders improved")
        verdict = "PASS" if imp == len(b) else "REJECT"
        print(f"    verdict: {verdict}"
              + ("" if verdict == "PASS" else
                 "  -- does not transfer; this is a parameter absorbing "
                 "per-folder error, not a property of the arm"))
        result[name + " LOFO"] = {
            "baseline_median_mm": float(np.median(b)),
            "transferred_median_mm": float(np.median(c)),
            "folds_improved": imp, "folds": len(b), "verdict": verdict}

    json.dump(result, open(args.out, "w"), indent=2, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
