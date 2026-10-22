#!/usr/bin/env python3
"""Render paired overlays for ``stationary_multicam.py`` results."""

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
from bundle_rims import project  # noqa: E402


def _draw_points(image, points, colour):
    visible = ((points[:, 0] >= 0) & (points[:, 0] < image.shape[1]) &
               (points[:, 1] >= 0) & (points[:, 1] < image.shape[0]))
    for x, y in points[visible].astype(int):
        cv2.circle(image, (x, y), 2, colour, -1, cv2.LINE_AA)


def _draw_pose(image, T, intrinsics, top_colour, floor_colour, outline_colour,
               line_width=2):
    rvec = cv2.Rodrigues(T[:3, :3])[0].ravel()
    tvec = T[:3, 3]
    points, normals, level, loop = board_model.build_levels()
    projected = project(points, rvec, tvec, intrinsics)
    visible_floor = board_model.visible(points, normals, level, loop, T[:3, :3], tvec)
    outline = project(board_model.outline(), rvec, tvec, intrinsics)
    cv2.polylines(image, [outline.astype(np.int32)], True, outline_colour,
                  line_width, cv2.LINE_AA)
    _draw_points(image, projected[level == 0], top_colour)
    _draw_points(image, projected[(level == 1) & visible_floor], floor_colour)


def _view_overlay(image_path, row, intrinsics, T_cam_board=None, joint=None,
                  preview_size=None):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"could not read {image_path}")
    T = np.asarray(row["T_cam_board"] if T_cam_board is None else T_cam_board)
    # Magenta is the locally image-refined pose.  In a calibrated run green and
    # cyan are the one shared rig pose that is actually reported downstream.
    # Their separation makes calibration error visible instead of hiding it.
    if T_cam_board is not None:
        _draw_pose(image, np.asarray(row["T_cam_board"]), intrinsics,
                   (255, 0, 255), (0, 165, 255), (255, 0, 255), 2)
    _draw_pose(image, T, intrinsics, (0, 255, 0), (255, 255, 0), (0, 255, 0), 2)
    pose_source = "joint rig pose" if T_cam_board is not None else "image fit"
    cv2.putText(image, f"{row['camera']}  {pose_source}  MAD {row['mad_px']:.2f}px  n={row['n']}",
                (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                cv2.LINE_AA)
    if T_cam_board is not None:
        per_camera = next((item for item in joint.get("per_camera", [])
                           if item["camera"] == row["camera"]), None)
        conditioning = joint.get("conditioning", {})
        condition_number = conditioning.get("condition_number_scaled")
        joint_text = "shared native-4K pose"
        if per_camera is not None:
            joint_text += f"  joint MAD {per_camera['mad_px']:.2f}px"
        if condition_number is not None:
            joint_text += f"  cond {condition_number:.1e}"
        cv2.putText(image, joint_text, (24, 82), cv2.FONT_HERSHEY_SIMPLEX,
                    0.70, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "green/cyan: shared rig pose   magenta/orange: local image refinement",
                    (24, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
                    cv2.LINE_AA)
    elif joint is not None and joint.get("status") == "unverified_calibration":
        cv2.putText(image, "UNVERIFIED RIG: local image fit only; shared pose withheld",
                    (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 80, 255), 2,
                    cv2.LINE_AA)
    if preview_size is not None:
        return cv2.resize(image, preview_size)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path,
                        default=HERE / "out" / "stationary_multicam.json")
    parser.add_argument("--data", type=Path, default=ROOT / "run_01")
    parser.add_argument("--outdir", type=Path, default=HERE / "out_stationary_overlays")
    args = parser.parse_args()

    fitted = json.loads(args.src.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)
    intrinsics = {camera: np.r_[values["K"][0][0], values["K"][1][1],
                                      values["K"][0][2], values["K"][1][2], values["D"]]
                  for camera, values in fitted["camera_intrinsics"].items()}
    by_capture = {}
    for row in fitted["view_fits"]:
        by_capture.setdefault(row["capture"], []).append(row)

    board_pose_rows = {row["capture"]: row for row in fitted["board_poses"]}
    camera_poses = {camera: np.asarray(T_reference_camera)
                    for camera, T_reference_camera
                    in fitted["camera_poses_in_reference"].items()}
    use_shared_pose = (fitted.get("calibration") is not None
                       or fitted.get("rig_source") is not None)
    for capture, rows in sorted(by_capture.items()):
        panels = []
        for row in sorted(rows, key=lambda r: r["camera"]):
            shared_T = None
            pose_row = board_pose_rows.get(capture)
            joint = pose_row.get("joint") if pose_row is not None else None
            is_verified = joint is not None and joint.get("status") == "verified"
            if use_shared_pose and is_verified:
                shared_T = (np.linalg.inv(camera_poses[row["camera"]]) @
                            np.asarray(pose_row["T_reference_board"]))
            image_path = args.data / capture / row["camera"] / "color.png"
            native = _view_overlay(image_path, row, intrinsics[row["camera"]],
                                   shared_T, joint)
            cv2.imwrite(str(args.outdir / f"{capture}_{row['camera']}_4k.png"), native)
            panels.append(cv2.resize(native, (960, 540)))
        panel = np.hstack(panels)
        pose_row = board_pose_rows.get(capture)
        joint = pose_row.get("joint") if pose_row is not None else None
        if not use_shared_pose:
            rig_label = "LOCAL IMAGE FITS"
            colour = (220, 220, 220)
        elif joint and joint.get("status") == "verified":
            rig_label = "VERIFIED RIG"
            colour = (80, 255, 80)
        else:
            rig_label = "UNVERIFIED RIG"
            colour = (0, 80, 255)
        cv2.putText(panel, f"{capture}: {rig_label}; green = top rims/outline; cyan = floor rims",
                    (24, panel.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    colour, 2, cv2.LINE_AA)
        cv2.imwrite(str(args.outdir / f"{capture}.png"), panel)
    print(f"wrote {len(by_capture)} paired overlays to {args.outdir}")


if __name__ == "__main__":
    main()
