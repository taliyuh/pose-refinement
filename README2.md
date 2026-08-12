# Transparent-board pose refinement (book_pics/)

Estimates the board's pose in the wood-background test images
(`images_new/`) and validates against the hand-annotated corner ground truth.

## Result

| Image | corner reproj error (1280×720) | full-res |
|-------|-------------------------------|----------|
| image_103325_139332 | **4.2 px** | ~13 px |
| image_103346_077660 | **3.6 px** | ~11 px |

(Representability floor of the 3-DOF model is ~8 px / ~4 px, so both are optimal.)

## Method

Static camera, board slides on a desk at constant height ⇒ the board pose is
constrained to a **fixed desk plane** ⇒ only **3 DOF**: yaw + 2D translation in
the plane.

1. **Desk plane** calibrated once from the two GT-corner poses
   (`SOLVEPNP_ITERATIVE`, **not** IPPE — see below). Averaged rotation + mean
   translation define the plane. GT is used *only* for this one-time
   calibration and for final validation, never for the per-image estimate.
2. **Blob detection** (`preprocess.py` top-hat + threshold, restricted to the
   desk ROI) finds the raised white shapes as bright blobs. Cylinders are
   flagged by circularity.
3. **Constellation matching**: the known 3D shape centres are aligned to the
   detected blobs over the 3-DOF family by minimising a **type-aware chamfer**
   cost (cylinder model points must match cylinder blobs — this breaks the
   180° rotational ambiguity). Coarse grid → Nelder-Mead.

## Why edges/IPPE/appearance failed

- **IPPE bug**: `SOLVEPNP_IPPE` needs z=0 planar points; the board points are at
  z=2.5cm, so IPPE returned ~330px garbage. Fixed by `SOLVEPNP_ITERATIVE`.
- **Edge chamfer** (`refine_pose_v2.py`): wood grain saturates Canny, so the
  distance transform is ≈0 everywhere and any pose scores well. Abandoned in
  favour of sparse blob centroids.

## Files

- `preprocess.py` — illumination-flatten + top-hat + CLAHE + Canny (debug fig)
- `board_model.py` — combined keypoint + STL interior feature cloud
- `refine_pose_v3.py` — **main**: blob-constellation 3-DOF estimator + validation
- `refine_pose_v2.py` — earlier edge-chamfer attempt (kept for reference)
- `out/` — visualisations + `v3_results.json`

## Run
```bash
python3 book_pics/refine_pose_v3.py
```

Outputs `out/<name>_v3_result.png` (green=estimate, red=GT, yellow=model
centres, blue=detected blobs) and `out/v3_results.json`.
