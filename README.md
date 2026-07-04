# Board Pose — Synthetic Dataset Generator

Generates synthetic images of a board on a checkerboard-calibrated desk, with BlenderProc. Output is HDF5 files containing color + segmentation maps, and JSON keypoint annotations.

## Setup

Requires the `ros2_ws` distrobox with BlenderProc installed:

```bash
distrobox enter ros2_ws
source .venv/bin/activate
```

## Generate data

```bash
blenderproc run generate_data.py
```

Output goes to `synthetic_dataset/`:
- `{0,1,2,...}.hdf5` — color images + segmaps per frame
- `annotations/frame_{n}.json` — 2D keypoint projections

Change `NUM_FRAMES` at the top of the script to generate more images.

## View an HDF5 image

```bash
python3 -c "
import h5py, numpy as np
from PIL import Image

f = h5py.File('synthetic_dataset/0.hdf5', 'r')
img = f['colors'][:]                              # shape: (480, 640, 3)
if img.max() <= 1.0:
    img = (img * 255).astype(np.uint8)
Image.fromarray(img).save('frame_0.png')
print('Saved frame_0.png')
f.close()
"
```

The HDF5 keys inside each file:
- `colors` — rendered RGB image
- `category_id_segmaps` — per-pixel category (0 = background/desk/cloth, 1 = board)
- `instance_segmaps` — per-pixel instance indices

These might be viewed using native bleanderproc method:

```bash
blenderproc vis hdf5 synthetic_dataset/0.hdf5
```

## ArUco localisation

The companion script `aruco_localise.py` runs ArUco-based hand-eye calibration using real camera captures.

## Key config values

| Variable | Meaning |
|----------|---------|
| `NUM_FRAMES` | How many synthetic frames to render |
| `SCALE_FACTOR` | Multiplier for the STL model (original is 32m wide) |
| `keypoints_3d` | Board keypoints in **scaled** world coordinates |
