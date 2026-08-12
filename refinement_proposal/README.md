# Improved Pose Refinement Pipeline (`refinement_proposal/`)

This directory contains the improved 3-DOF transparent board pose estimation pipeline.

## Files

* **`pose_lib_improved.py`**: The core pose estimation library, self-contained and updated to use the exact shape centers from the CAD model coordinates.
* **`aruco_estimate_improved.py`**: Runs ArUco marker detection (with sub-pixel refinement) and estimates/visualizes the board pose.
* **`evaluate_improved.py`**: Evaluates the corner reprojection error of the improved pipeline on all ground-truth annotated wood images.

---

## Instructions

First, enter the container and activate the virtual environment:
```bash
distrobox enter ros-dev
source .venv/bin/activate
```

### 1. Run Pose Estimation
To run pose estimation on the default directory (`images/images_wood`):
```bash
python3 refinement_proposal/aruco_estimate_improved.py
```

To run on a specific directory:
```bash
python3 refinement_proposal/aruco_estimate_improved.py path/to/images_dir
```

Output visualization images will be saved under `refinement_proposal/out/`.

### 2. Run Evaluation
To calculate the average corner reprojection error across all annotated images:
```bash
python3 refinement_proposal/evaluate_improved.py
```
This automatically loads annotations from `images/ground truth.txt` and evaluates against `images/images_wood`.
