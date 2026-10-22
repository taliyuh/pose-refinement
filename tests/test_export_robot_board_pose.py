import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "boardpose"))

import export_robot_board_pose as exporter  # noqa: E402


class ExportVerificationTests(unittest.TestCase):
    def _refinement(self, status):
        return {
            "frame": "mega_camera",
            "board_poses": [{
                "capture": "capture_0001",
                "T_reference_board": np.eye(4).tolist(),
                "joint": {"status": status},
            }],
        }

    def test_rejects_non_verified_refinement(self):
        with self.assertRaisesRegex(ValueError, "only a verified"):
            exporter._board_fit(self._refinement("unverified_calibration"), None)

    def test_accepts_verified_refinement(self):
        frame, capture, matrix, _ = exporter._board_fit(
            self._refinement("verified"), None)
        self.assertEqual(frame, "mega_camera")
        self.assertEqual(capture, "capture_0001")
        np.testing.assert_allclose(matrix, np.eye(4))

    def test_selects_camera_from_joint_calibration(self):
        calibration = {
            "schema_version": 2,
            "base_frame": "base",
            "camera_frames": {"mega": "mega_camera", "bolt": "bolt_camera"},
            "T_base_camera": {
                "mega": np.eye(4).tolist(),
                "bolt": (np.diag([1.0, -1.0, -1.0, 1.0])).tolist(),
            },
        }
        base, frame, matrix = exporter._calibration_transform(calibration, "bolt")
        self.assertEqual((base, frame), ("base", "bolt_camera"))
        np.testing.assert_allclose(matrix, calibration["T_base_camera"]["bolt"])

    def test_requires_camera_when_joint_calibration_has_multiple_cameras(self):
        calibration = {
            "schema_version": 2,
            "base_frame": "base",
            "camera_frames": {"mega": "mega_camera", "bolt": "bolt_camera"},
            "T_base_camera": {"mega": np.eye(4).tolist(), "bolt": np.eye(4).tolist()},
        }
        with self.assertRaisesRegex(ValueError, "multiple cameras"):
            exporter._calibration_transform(calibration)
