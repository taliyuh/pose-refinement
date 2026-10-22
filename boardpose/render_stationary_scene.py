#!/usr/bin/env python3
"""Render an isometric 3D overview of a stationary camera-rig fit."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import board_model  # noqa: E402

WIDTH, HEIGHT = 1400, 1000


def _transform(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def _scene_projection(points):
    """Return a stable orthographic isometric projection for metric points."""
    centre = points.mean(axis=0)
    screen_x = np.array([0.707, -0.707, 0.0])
    screen_y = np.array([0.408, 0.408, -0.816])
    projected = np.stack([(points - centre) @ screen_x,
                          (points - centre) @ screen_y], axis=1)
    span = max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1]), 0.1)
    scale = 0.75 * min(WIDTH, HEIGHT) / span

    def project(scene_points):
        local = scene_points - centre
        pixels = np.stack([local @ screen_x, local @ screen_y], axis=1) * scale
        pixels[:, 0] += WIDTH / 2
        pixels[:, 1] += HEIGHT / 2
        return pixels.astype(np.int32)

    return project


def _draw_axes(image, project, transform, label, colour):
    origin = transform[:3, 3]
    axes = np.vstack([origin, origin + transform[:3, 0] * 0.1,
                      origin + transform[:3, 1] * 0.1,
                      origin + transform[:3, 2] * 0.1])
    pixels = project(axes)
    cv2.circle(image, tuple(pixels[0]), 7, colour, -1, cv2.LINE_AA)
    for endpoint, axis_colour in zip(pixels[1:], ((0, 0, 255), (0, 255, 0),
                                                   (255, 0, 0))):
        cv2.arrowedLine(image, tuple(pixels[0]), tuple(endpoint), axis_colour, 2,
                        cv2.LINE_AA, tipLength=0.15)
    cv2.putText(image, label, tuple(pixels[0] + [10, -10]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path,
                        default=HERE / "out" / "stationary_multicam.json")
    parser.add_argument("--out", type=Path,
                        default=HERE / "out_stationary_overlays" / "scene_3d.png")
    args = parser.parse_args()

    fitted = json.loads(args.src.read_text())
    board_outline = board_model.outline()
    camera_poses = {name: np.asarray(pose)
                    for name, pose in fitted["camera_poses_in_reference"].items()}
    board_poses = [(record["capture"], np.asarray(record["T_reference_board"]))
                   for record in fitted["board_poses"]]

    scene_points = [pose[:3, 3] for pose in camera_poses.values()]
    for _, pose in board_poses:
        scene_points.extend(_transform(board_outline, pose))
    project = _scene_projection(np.asarray(scene_points))
    image = np.full((HEIGHT, WIDTH, 3), 28, dtype=np.uint8)

    # Each ray joins a camera centre to a fitted board centre. In a valid rig,
    # both cameras should point at the same board rather than in unrelated
    # directions.
    for _, board_pose in board_poses:
        for camera_pose in camera_poses.values():
            ray = project(np.asarray([camera_pose[:3, 3], board_pose[:3, 3]]))
            cv2.line(image, tuple(ray[0]), tuple(ray[1]), (75, 75, 75), 1,
                     cv2.LINE_AA)

    for capture, pose in board_poses:
        polygon = project(_transform(board_outline, pose))
        cv2.fillConvexPoly(image, polygon, (80, 120, 80), cv2.LINE_AA)
        cv2.polylines(image, [polygon], True, (80, 255, 80), 3, cv2.LINE_AA)
        centre = project(np.asarray([pose[:3, 3]]))[0]
        cv2.putText(image, capture, tuple(centre + [10, 20]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2,
                    cv2.LINE_AA)

    for camera, pose in camera_poses.items():
        _draw_axes(image, project, pose, camera, (0, 220, 255))

    cv2.putText(image, f"3D stationary-rig view ({fitted['frame']})", (30, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Boards: green   Cameras: yellow   rays: grey   axes: X red, Y green, Z blue",
                (30, HEIGHT - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (220, 220, 220), 2, cv2.LINE_AA)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), image)
    print(f"wrote 3D rig scene to {args.out}")


if __name__ == "__main__":
    main()
