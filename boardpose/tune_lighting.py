#!/usr/bin/env python3
"""
tune_lighting.py - score a single frame on the quantities that actually limit
the fit, so the lamps can be aimed in a feedback loop instead of by argument.

Needs no pose and no seed: it finds the pockets with a Hough transform, so it
runs on a frame straight out of the camera.

READ THIS BEFORE TRUSTING ANY NUMBER BELOW
------------------------------------------
Two of these measures are arithmetic and two are hypotheses.

ARITHMETIC, so they can be acted on directly. A pixel at 255 has zero gradient,
so a clipped rim is DELETED, not merely bright -- the subpixel search has nothing
to lock onto. A pocket floor crushed toward 0 has no detectable rim at all. Both
are facts about the edge search, not statistical claims, and both are flagged
hard below.

HYPOTHESES, so they are reported without thresholds. Rim contrast and azimuthal
uniformity are *believed* to drive the residual: the weakest quartile of edges in
the current 47 views sits at 0.81 px MAD against 0.36 px for the strongest, and a
rim lit brightly on one side should have its gradient peak pulled that way, which
would be a directional bias no rigid 6-DoF pose can absorb -- a candidate
explanation for the -0.3..-0.8 px mean residual present in every view.

That belief is UNVALIDATED, and this dataset cannot validate it. Scoring all 47
existing views and correlating against their fitted residual gives, controlling
for standoff distance:

    sharpness   +0.53      azimuthal uniformity   -0.42
    contrast    -0.20      pocket floor level     -0.15
    clipping    -0.20

Half of those have the wrong sign, sharpness most strongly. The reason is
structural rather than surprising: all 47 views were shot under ONE lighting
setup, so the between-view variation in these numbers is view geometry and
obliquity, not illumination. A within-setup null result says nothing about what
happens when the setup changes. (Compare the README's "Superseded files" note --
heuristic 2D quality metrics have already been found to sit at chance here once
before, for a related reason.)

So: use this tool COMPARATIVELY, across lighting changes, which is what --watch
is for. Move one lamp, capture, see which way the numbers go. Do not read the
absolute values as a quality score. Baselines from the existing 47 views are
printed alongside each number so there is something to compare against.

Usage:
    tune_lighting.py FRAME.png [FRAME2.png ...]
    tune_lighting.py --watch DIR     # rescore the newest frame as it appears
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Hough is run on a 4x downscale. Hole diameters are 12-22 mm; at a ~480 mm
# standoff with fx ~ 2251 px that is 56-103 px at 4K, so 14-26 px here.
DOWN = 4
RMIN, RMAX = 6, 15


def candidates(gray, cap=40):
    """Rough pocket circles straight from Hough, at full-resolution scale.

    Capped, because refine() costs ~0.15 s each and nothing upstream bounds how
    many circles Hough returns -- an unbounded candidate list is how this turned
    a 3 s call into an apparent hang. Hough returns them in accumulator order,
    so the cap keeps the strongest.
    """
    small = cv2.resize(gray, None, fx=1.0 / DOWN, fy=1.0 / DOWN,
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.0)
    c = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1, minDist=RMIN * 2,
                         param1=80, param2=12, minRadius=RMIN, maxRadius=RMAX)
    if c is None:
        return np.zeros((0, 3))
    return c[0][:cap] * DOWN


def find_pockets(gray):
    """Locate pocket mouths. Returns Nx3 (cx, cy, r) at full resolution.

    Hough alone is not good enough on a transparent board -- run loose, it finds
    about 8 of the 20 rims with radius errors up to 14 px, plus false positives.
    So it is used only as a candidate generator; every candidate is then refined
    against the actual radial gradient and dropped if it does not look like a
    rim. That makes the tool self-correcting: it does not need to find all 20,
    it needs the ones it reports to be real, because the numbers are read
    comparatively (did moving that lamp help?) rather than absolutely.
    """
    out = []
    for cx, cy, r in candidates(gray):
        best = refine(gray, cx, cy, r)
        if best is not None:
            out.append(best)
    if not out:
        return np.zeros((0, 3))

    # drop duplicates that converged onto the same rim, then keep the 20 strongest
    # -- the board has exactly 20 pockets, so anything beyond that is a false
    # positive and would only add noise to a comparison between lamp positions
    out.sort(key=lambda t: -t[3])
    keep = []
    for cand in out:
        if all(np.hypot(cand[0] - k[0], cand[1] - k[1]) > 0.8 * k[2] for k in keep):
            keep.append(cand)
        if len(keep) == 20:
            break
    return np.array([k[:3] for k in keep])


def refine(gray, cx, cy, r):
    """Slide (cx, cy, r) onto the true rim by maximising the radial gradient.

    Hough's centres land 2-12 px out and its radii up to 14 px out, so both are
    searched. Returns (cx, cy, r, strength), or None if the candidate never
    looks like a circular step -- which is how false positives are thrown out.
    """
    def strength(x, y, rr):
        s, _, _ = rim_profile(gray, x, y, rr, nrad=9, half=0.10)
        ok = ~np.isnan(s)
        return np.median(s[ok]) if ok.sum() >= 12 else -1.0

    step, span = 4.0, 0.35
    best = (cx, cy, r, strength(cx, cy, r))
    for _ in range(3):
        for radius in r * np.linspace(1 - span, 1 + span, 13):
            best = _best_nearby_centre(strength, cx, cy, radius, step, best)
        cx, cy, r = best[:3]
        step *= 0.45
        span *= 0.45

    if best[3] < MIN_RIM_STRENGTH:
        return None
    return best


MIN_RIM_STRENGTH = 3.0


def _best_nearby_centre(strength, cx, cy, radius, step, best):
    """Best of the 3x3 centre offsets at one radius, against `best` so far."""
    for dx in (-step, 0.0, step):
        for dy in (-step, 0.0, step):
            score = strength(cx + dx, cy + dy, radius)
            if score > best[3]:
                best = (cx + dx, cy + dy, radius, score)
    return best


def rim_profile(gray, cx, cy, r, nsect=16, nrad=21, half=0.28):
    """Sample the radial intensity gradient across one rim, by sector.

    Returns (contrast per sector, interior mean, exterior mean). Contrast is the
    peak absolute radial derivative found within +-`half` of the nominal radius,
    which is where the rim is regardless of small centring errors.
    """
    h, w = gray.shape
    th = np.linspace(0, 2 * np.pi, nsect, endpoint=False)
    rad = np.linspace((1 - half) * r, (1 + half) * r, nrad)
    out = np.empty(nsect)
    for i, a in enumerate(th):
        xs = cx + rad * np.cos(a)
        ys = cy + rad * np.sin(a)
        if xs.min() < 1 or ys.min() < 1 or xs.max() > w - 2 or ys.max() > h - 2:
            out[i] = np.nan
            continue
        v = cv2.remap(gray, xs.astype(np.float32).reshape(-1, 1),
                      ys.astype(np.float32).reshape(-1, 1),
                      cv2.INTER_LINEAR).ravel().astype(np.float64)
        out[i] = np.abs(np.diff(v)).max() / (rad[1] - rad[0])

    yy, xx = np.mgrid[max(0, int(cy - r)):min(h, int(cy + r) + 1),
                      max(0, int(cx - r)):min(w, int(cx + r) + 1)]
    d = np.hypot(xx - cx, yy - cy)
    patch = gray[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    inner = patch[d <= 0.6 * r]
    ring = patch[(d > 1.15 * r) & (d < 1.45 * r)]
    return out, (inner.mean() if inner.size else np.nan), \
                (ring.mean() if ring.size else np.nan)


def sharpness(gray):
    """Variance of Laplacian over the centre crop.

    CONTENT-DEPENDENT, so do not read it as focus. It scales with how much
    high-contrast material happens to be in the crop: on two frames of the same
    board at the same focus it read 54.2 and 94.4, a 1.74x spread, while their
    busy-pixel fractions were 6.20% and 11.43%, a 1.84x spread -- i.e. it was
    measuring clutter. Frame the smooth transparent board tightly and the number
    collapses with focus unchanged. Use rise_width() to judge focus.
    """
    h, w = gray.shape
    c = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return float(cv2.Laplacian(c, cv2.CV_64F).var())


def rise_width(gray, cx, cy, r, nsect=24):
    """Median 10-90% intensity rise distance across one rim, in pixels.

    This is the honest focus measure: a rise width is a property of the optics
    and the sampling, independent of how much contrast the edge has and of how
    much else is in the frame. On the two frames above it gave 1.79 and 1.71 px
    -- a 5% difference -- correctly reporting that focus was unchanged.
    """
    out = []
    rad = np.linspace(0.80 * r, 1.20 * r, 81)
    step = rad[1] - rad[0]
    h, w = gray.shape
    for a in np.linspace(0, 2 * np.pi, nsect, endpoint=False):
        xs = cx + rad * np.cos(a)
        ys = cy + rad * np.sin(a)
        if xs.min() < 1 or ys.min() < 1 or xs.max() > w - 2 or ys.max() > h - 2:
            continue
        v = cv2.remap(gray, xs.astype(np.float32).reshape(-1, 1),
                      ys.astype(np.float32).reshape(-1, 1),
                      cv2.INTER_LINEAR).ravel().astype(np.float64)
        k = int(np.abs(np.diff(v)).argmax())
        seg = v[max(0, k - 12):min(len(v), k + 13)]
        if seg.size < 8:
            continue
        a0, a1 = seg[0], seg[-1]
        if abs(a1 - a0) < 12:          # no real step here, skip
            continue
        lo, hi = sorted([a0 + 0.1 * (a1 - a0), a0 + 0.9 * (a1 - a0)])
        idx = np.nonzero((seg - lo) * (seg - hi) <= 0)[0]
        if idx.size < 1:
            continue
        out.append((idx.max() - idx.min() + 1) * step)
    return float(np.median(out)) if out else float("nan")


def focus(gray, circles=None):
    """Frame-level focus: median rim rise width in px. NaN if no rims found.

    Deliberately uses RAW Hough circles, not refined ones. rise_width() locates
    the steepest point along each radial ray itself, so a circle that is a few px
    out is good enough -- and skipping refine() takes this from 2.2-3.4 s to
    under 0.1 s. Checked against the refined version on six frames: agreement
    within +/-0.15 px (mean 0.08), while still tracking defocus monotonically
    (sigma 0/0.8/1.6/2.5 blur -> 2.06/2.46/2.72/3.04 px). Cheap enough to run on
    every capture, which is the whole point.
    """
    if circles is None:
        circles = candidates(gray, cap=24)
    widths = []
    for cx, cy, r in circles:
        width = rise_width(gray, cx, cy, r)
        if np.isfinite(width):
            widths.append(width)

    if not widths:
        return float("nan")
    return float(np.median(widths))


def score(path, verbose=True):
    img = cv2.imread(str(path))
    if img is None:
        print(f"  cannot read {path}")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    hi = float((gray >= 254).mean()) * 100
    lo = float((gray <= 1).mean()) * 100
    sh = sharpness(gray)

    circles = find_pockets(gray)
    rows = []
    for cx, cy, r in circles:
        sect, inner, ring = rim_profile(gray, cx, cy, r)
        ok = ~np.isnan(sect)
        if ok.sum() < 12:
            continue
        s = sect[ok]
        p90, p10 = np.percentile(s, [90, 10])
        rows.append((np.median(s), p90 / max(p10, 1e-6), inner, ring, r))

    # Medians over the existing 47 views, as a reference point. These are what the
    # OLD lighting produced, not a target -- the whole aim is to beat them.
    print(f"\n{Path(path).name}   {w}x{h}"
          f"{'':>18}old 47-view median")
    print(f"  sharpness (var Laplacian)   {sh:8.1f}          {35.3:8.1f}"
          f"   <- CLUTTER, not focus")
    print(f"  clipped high (>=254)        {hi:8.3f} %        {0.37:8.3f} %"
          f"   <- HARD LIMIT, want 0.000")
    print(f"  clipped low  (<=1)          {lo:8.3f} %        {0.00:8.3f} %"
          f"   <- HARD LIMIT, want 0.000")

    if not rows:
        print("  no pockets found -- is the board in frame and roughly nadir?")
        return None

    a = np.array(rows)
    contrast, unif, inner, ring = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    q = np.percentile(contrast, [10, 50])
    print(f"  pockets locked onto         {len(rows):8d}          "
          f"{'(of 20)':>8s}")
    fw = focus(gray, circles)
    print(f"  FOCUS: rim 10-90% rise      {fw:8.2f} px      {'~1.75':>8s} px"
          f"   <- the real focus number")
    print(f"  rim contrast   median       {q[1]:8.1f} /px      {14.7:8.1f} /px")
    print(f"                 weakest 10%  {q[0]:8.1f} /px      {11.3:8.1f} /px")
    print(f"  azimuthal uniformity p90/p10{np.median(unif):8.2f} x       "
          f"{6.58:8.2f} x")
    print(f"  pocket floor level          {np.median(inner):8.1f}          "
          f"{186.0:8.1f}")
    print(f"  top face level              {np.median(ring):8.1f}")

    # Only the arithmetic failures are flagged. Contrast and uniformity are
    # deliberately NOT thresholded -- see the module docstring; their relationship
    # to the fitted residual is unvalidated and this dataset cannot validate it.
    flags = []
    if hi > 0.001:
        flags.append("HIGHLIGHT CLIPPING - shorten the exposure or diffuse the "
                     "lamps. A pixel at 255 has zero gradient, so a clipped rim "
                     "is deleted rather than merely bright.")
    if lo > 0.001:
        flags.append("SHADOW CLIPPING - same problem at the other end.")
    if np.median(inner) < 20:
        flags.append("POCKET FLOORS NEAR BLACK - no rim to detect down there. "
                     "Light needs >61 deg elevation to reach the floor of a "
                     "12 mm pocket 22 mm deep; white card under the board is the "
                     "other route. Tilt is the DoF that pays for this.")
    if len(rows) < 10:
        flags.append(f"only {len(rows)} pockets locked onto - the comparison "
                     "across lamp positions will be noisy; get the board more "
                     "square to the camera before tuning.")
    for f in flags:
        print(f"  ** {f}")
    if not flags:
        print("  ** no hard failure. Now move ONE lamp and re-read the "
              "contrast / uniformity lines; they are meaningful as a direction "
              "of change, not as an absolute score.")
    return sh, hi, q[1], np.median(unif), np.median(inner)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", nargs="*")
    ap.add_argument("--watch", metavar="DIR",
                    help="rescore the newest color.png under DIR as it appears")
    args = ap.parse_args()

    if args.watch:
        root = Path(args.watch)
        seen = None
        print(f"watching {root} -- move a lamp, capture, read the numbers. ^C to stop.")
        while True:
            f = sorted(root.rglob("color.png"), key=lambda p: p.stat().st_mtime)
            if f and f[-1] != seen:
                seen = f[-1]
                score(seen)
            time.sleep(1.0)

    if not args.frames:
        ap.error("give one or more frames, or --watch DIR")
    for f in args.frames:
        score(f)


if __name__ == "__main__":
    sys.exit(main())
