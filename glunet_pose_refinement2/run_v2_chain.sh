#!/usr/bin/env bash
# run_v2_chain.sh - wait for the v2 rim fit, then run everything downstream.
#
# Launched detached while fit_rims6.py is still running. It blocks on that PID,
# refuses to proceed on a partial fit, and stops the chain at the first failing
# step so a bad bundle never reaches the cache or the kinematic calibration.
#
#   usage: run_v2_chain.sh <pid-of-fit_rims6> <fit-log>

set -u
FIT_PID="${1:?pid required}"
FIT_LOG="${2:?fit log required}"

D=/var/home/bartek/board_pose_obtain
PY="$D/.venv/bin/python -u"
G="$D/glunet_pose_refinement2"
FIT="$G/out/rims6_v2.json"
BUNDLE="$G/out/bundle_rims6_v3.json"

say() { echo "[$(date +%H:%M:%S)] $*"; }
step() {
    say "--- $1 ---"
    shift
    if ! "$@"; then
        say "FAILED: $*"
        say "CHAIN ABORTED"
        exit 1
    fi
}

say "waiting for fit_rims6.py (pid $FIT_PID)"
while kill -0 "$FIT_PID" 2>/dev/null; do sleep 30; done
say "fit process exited"

# Did it finish, or was it killed? main() prints the summary only on a clean
# run, and the resume logic means a partial file looks perfectly valid on disk.
if ! grep -q "^wrote " "$FIT_LOG"; then
    say "fit did not reach its summary line -- it was killed or crashed."
    say "the partial results are in $FIT (rerun fit_rims6.py to resume)"
    say "CHAIN ABORTED"
    exit 1
fi
N=$($PY -c "import json;print(len(json.load(open('$FIT'))))")
say "fit complete: $N views"

step "bundle adjustment" $PY "$G/bundle_rims.py" --init "$FIT" \
    --rim-bias b0 --out "$BUNDLE"
step "overlays" $PY "$G/render_fit.py" --src "$BUNDLE" --outdir "$G/out_overlays6_v2"
step "cached_views" $PY "$G/update_cached_from_bundle.py" --bundle "$BUNDLE" --apply
step "kinematic calibration" $PY "$G/kinematic_calib.py" --src "$BUNDLE" \
    --out "$G/out/kinematic_calib_v3.json"

# The multi-view board pose is what downstream actually consumes, and it is the
# only step whose output is a pose rather than a calibration. Its split-half
# number is also the pipeline's honest end-to-end precision, so a regression
# here is visible without any ground truth.
step "multi-view board pose (all folders)" $PY "$G/eval_multiview.py" \
    --bundle "$BUNDLE" --calib "$G/out/kinematic_calib_v3.json" \
    --out "$G/out/eval_multiview.json"

say "--- v1 vs v2 ---"
$PY - <<'EOF'
import json
import numpy as np
G = "/var/home/bartek/board_pose_obtain/glunet_pose_refinement2"


def mads(p):
    d = json.load(open(p))
    v = d["views"] if isinstance(d, dict) else d
    return np.array([r["mad_px"] for r in v])


for tag, a, b in [("per-view fit", "rims6.json", "rims6_v2.json"),
                  ("bundle", "bundle_rims6.json", "bundle_rims6_v2.json")]:
    try:
        x, y = mads(f"{G}/out/{a}"), mads(f"{G}/out/{b}")
    except FileNotFoundError as e:
        print(f"{tag}: {e}")
        continue
    print(f"{tag:14} v1 n={len(x):3d} median {np.median(x):.3f} px "
          f"pass<1 {(x < 1).sum():3d}   ->   "
          f"v2 n={len(y):3d} median {np.median(y):.3f} px "
          f"pass<1 {(y < 1).sum():3d}")

for tag, f in [("v1", "kinematic_calib.json"), ("v2", "kinematic_calib_v2.json")]:
    try:
        k = json.load(open(f"{G}/out/{f}"))
    except FileNotFoundError:
        continue
    print(f"kinematic {tag}: " + "  ".join(
        f"{n}={v.get('median_mm', float('nan')):.3f}mm"
        for n, v in k.items() if isinstance(v, dict) and "median_mm" in v))
    cv = k.get("cross_validation")
    if cv:
        print(f"  LOFO {tag}: {cv['baseline_median_mm']:.3f} -> "
              f"{cv['transferred_median_mm']:.3f} mm, "
              f"{cv['folds_improved']}/{cv['folds']} folds improved")
EOF

say "CHAIN COMPLETE"
