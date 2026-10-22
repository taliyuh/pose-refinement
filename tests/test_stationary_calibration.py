import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "boardpose"))

try:
    import cv2  # noqa: F401
    import scipy  # noqa: F401
    import trimesh  # noqa: F401
except ImportError as error:
    RUNTIME_ERROR = str(error)
    RUNTIME_AVAILABLE = False
else:
    RUNTIME_ERROR = ""
    RUNTIME_AVAILABLE = True
    import calibrate_stationary_rig as rig  # noqa: E402
    import fit_rims6  # noqa: E402
    import stationary_multicam as stationary  # noqa: E402


@unittest.skipUnless(RUNTIME_AVAILABLE, f"runtime dependencies unavailable: {RUNTIME_ERROR}")
class CalibrationLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.info_paths = {}
        self.intrinsics = {}
        for camera in ("mega", "bolt"):
            path = self.directory / f"{camera}_camera_info.json"
            path.write_text(json.dumps({
                "width": 3840,
                "height": 2160,
                "k": [1000, 0, 1920, 0, 1000, 1080, 0, 0, 1],
                "d": [0, 0, 0, 0, 0],
            }))
            self.info_paths[camera] = path
            self.intrinsics[camera] = stationary._camera_intrinsics(path)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _calibration(self):
        cameras = ("mega", "bolt")
        return {
            "version": stationary.SUPPORTED_CALIBRATION_VERSION,
            "calibration_status": "accepted",
            "reference_camera": "mega",
            "camera_poses_in_reference": {
                "mega": np.eye(4).tolist(),
                "bolt": np.eye(4).tolist(),
            },
            "camera_intrinsics": {
                camera: stationary._camera_intrinsics_provenance(
                    self.info_paths[camera], self.intrinsics[camera])
                for camera in cameras
            },
            "rim_bias_model": {
                "version": 1,
                "per_camera": {
                    camera: {"names": ["b0_top", "b0_floor"], "bias_m": [0, 0]}
                    for camera in cameras
                },
            },
        }

    def _load(self, calibration):
        path = self.directory / "rig.json"
        path.write_text(json.dumps(calibration))
        return stationary._load_calibration(
            path, "mega", ["mega", "bolt"], self.intrinsics, self.info_paths)

    def _joint_calibration(self):
        cameras = ("mega", "bolt")
        return {
            "schema_version": 2,
            "calibration_type": "joint_robot_camera_handeye",
            "base_frame": "base",
            "units": "m",
            "camera_frames": {camera: f"{camera}_camera" for camera in cameras},
            "T_base_camera": {camera: np.eye(4).tolist() for camera in cameras},
            "T_tcp_tag": np.eye(4).tolist(),
            "quality": {
                "samples_total": 10,
                "samples_used": 10,
                "per_camera": {
                    camera: {
                        "samples_total": 5,
                        "samples_used": 5,
                        "mean_reprojection_rms_px": 0.2,
                        "max_reprojection_rms_px": 0.4,
                    }
                    for camera in cameras
                },
            },
            "validation": {
                "status": "completed",
                "method": "leave_one_pose_out",
                "observation_model": "direct_distorted_image_corners",
                "samples_evaluated": 10,
                "failures": [],
                "undetected_indices": {camera: [] for camera in cameras},
            },
            "provenance": {
                "dataset": "/tmp/joint_capture",
                "camera_names": list(cameras),
            },
        }

    def test_loads_only_current_accepted_calibration(self):
        _, poses = self._load(self._calibration())
        np.testing.assert_allclose(poses["mega"], np.eye(4))

    def test_loads_joint_handeye_artifact_for_each_camera(self):
        path = self.directory / "joint.json"
        path.write_text(json.dumps(self._joint_calibration()))
        _, provenance = stationary._load_camera_base_calibrations(
            [f"mega={path}", f"bolt={path}"], "mega", ["mega", "bolt"])
        self.assertEqual(provenance["mega"]["calibration_schema_version"], 2)
        self.assertEqual(provenance["bolt"]["calibration_schema_version"], 2)

    def test_loads_one_camera_from_joint_handeye_artifact(self):
        path = self.directory / "joint_single_camera.json"
        path.write_text(json.dumps(self._joint_calibration()))
        poses, provenance = stationary._load_camera_base_calibrations(
            [f"bolt={path}"], "bolt", ["bolt"])
        np.testing.assert_allclose(poses["bolt"], np.eye(4))
        self.assertEqual(provenance["bolt"]["calibration_schema_version"], 2)

    def test_reports_poor_joint_corner_quality_without_blocking_pipeline(self):
        calibration = self._joint_calibration()
        for camera in ("mega", "bolt"):
            calibration["quality"]["per_camera"][camera].update({
                "direct_corner_rms_px": 9.0,
                "direct_corner_max_rms_px": 12.0,
            })
        path = self.directory / "joint_poor_quality.json"
        path.write_text(json.dumps(calibration))

        output = io.StringIO()
        with redirect_stdout(output):
            poses, _ = stationary._load_camera_base_calibrations(
                [f"mega={path}", f"bolt={path}"], "mega", ["mega", "bolt"])

        np.testing.assert_allclose(poses["mega"], np.eye(4))
        self.assertIn("quality.direct_corner_rms_px=9", output.getvalue())

    def test_rejects_incomplete_joint_handeye_validation(self):
        calibration = self._joint_calibration()
        calibration["validation"]["status"] = "incomplete"
        path = self.directory / "joint_incomplete.json"
        path.write_text(json.dumps(calibration))
        with self.assertRaisesRegex(ValueError, "validation status"):
            stationary._load_camera_base_calibrations(
                [f"mega={path}", f"bolt={path}"], "mega", ["mega", "bolt"])

    def test_rejects_missing_acceptance_status(self):
        calibration = self._calibration()
        del calibration["calibration_status"]
        with self.assertRaisesRegex(ValueError, "calibration_status"):
            self._load(calibration)

    def test_rejects_older_calibration_schema(self):
        calibration = self._calibration()
        calibration["version"] -= 1
        with self.assertRaisesRegex(ValueError, "unsupported calibration version"):
            self._load(calibration)

    def test_rejects_non_se3_pose_and_intrinsics_mismatch(self):
        calibration = self._calibration()
        calibration["camera_poses_in_reference"]["bolt"][3][3] = 0
        with self.assertRaisesRegex(ValueError, "last row"):
            self._load(calibration)

        calibration = self._calibration()
        calibration["camera_intrinsics"]["bolt"]["K"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "bolt K differs"):
            self._load(calibration)

    def test_rejects_missing_or_out_of_bounds_rim_bias_model(self):
        calibration = self._calibration()
        del calibration["rim_bias_model"]
        with self.assertRaisesRegex(ValueError, "rim_bias_model"):
            self._load(calibration)

        calibration = self._calibration()
        calibration["rim_bias_model"]["per_camera"]["bolt"]["bias_m"] = [0.0006, 0]
        with self.assertRaisesRegex(ValueError, "0.5 mm"):
            self._load(calibration)

    def test_transform_recalibration_prior_requires_current_camera_binding(self):
        calibration = self._calibration()
        prior_path = self.directory / "prior.json"
        prior_path.write_text(json.dumps(calibration))
        _, poses, biases, provenance = rig._load_transform_recalibration_prior(
            prior_path, "mega", ["mega", "bolt"], self.intrinsics, self.info_paths)
        np.testing.assert_allclose(poses["bolt"], np.eye(4))
        np.testing.assert_allclose(biases["mega"], np.zeros(2))
        self.assertEqual(provenance["path"], str(prior_path.resolve()))
        self.assertEqual(len(provenance["sha256"]), 64)

        calibration["camera_intrinsics"]["bolt"]["D"][0] = 0.1
        prior_path.write_text(json.dumps(calibration))
        with self.assertRaisesRegex(ValueError, "bolt D differs"):
            rig._load_transform_recalibration_prior(
                prior_path, "mega", ["mega", "bolt"], self.intrinsics,
                self.info_paths)

        calibration = self._calibration()
        calibration["camera_poses_in_reference"]["extra"] = np.eye(4).tolist()
        prior_path.write_text(json.dumps(calibration))
        with self.assertRaisesRegex(ValueError, "do not exactly match"):
            rig._load_transform_recalibration_prior(
                prior_path, "mega", ["mega", "bolt"], self.intrinsics,
                self.info_paths)


@unittest.skipUnless(RUNTIME_AVAILABLE, f"runtime dependencies unavailable: {RUNTIME_ERROR}")
class CalibrationGateTests(unittest.TestCase):
    def test_capture_timing_reports_loose_camera_pair_without_rejecting(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "capture_0001"
            capture.mkdir()
            (capture / "capture.json").write_text(json.dumps({
                "camera_color_span_ms": 75.1,
            }))

            self.assertEqual(stationary._validate_capture_timing([capture], 75.0), 75.1)

    def test_provisional_transform_is_not_verified(self):
        status = stationary._provisional_joint_status()
        self.assertEqual(status["status"], "provisional_in_session_estimate")
        self.assertNotIn("verified", status["status"])

    def test_single_reference_camera_is_verified(self):
        status = stationary._single_camera_joint_status()
        self.assertEqual(status["status"], "verified")
        self.assertEqual(status["method"], "single_reference_camera")

    def test_single_camera_with_handeye_uses_direct_reference_pose(self):
        rows = [{
            "capture": "capture_0001",
            "camera": "bolt",
            "T_cam_board": np.eye(4).tolist(),
        }]
        board_poses = stationary._single_camera_board_poses(
            rows, {"bolt": np.eye(4)}, "bolt")

        self.assertEqual(len(board_poses), 1)
        self.assertEqual(board_poses[0]["joint"]["status"], "verified")
        self.assertEqual(board_poses[0]["joint"]["method"],
                         "single_reference_camera")
        np.testing.assert_allclose(
            board_poses[0]["T_reference_board"], np.eye(4))

    def test_fixed_rig_one_synchronized_pair_can_be_verified(self):
        self.assertTrue(stationary._fixed_rig_is_verified(
            [("capture_0001", np.eye(4))], 0.4, 0.7))

    def test_fixed_rig_requires_an_observed_pair_within_limit(self):
        self.assertFalse(stationary._fixed_rig_is_verified([], None, 0.7))
        self.assertFalse(stationary._fixed_rig_is_verified(
            [("capture_0001", np.eye(4))], 0.71, 0.7))

    def test_rim_bundle_checks_requested_minimum_after_rejections(self):
        transform, diagnostics = rig._bundle_transform_from_synchronized_rims(
            [], [], "mega", "bolt",
            {"mega": (np.eye(3), np.zeros(5)), "bolt": (np.eye(3), np.zeros(5))},
            np.eye(4), set(), 0.7, 1.0, 4,
            {"mega": np.zeros(2), "bolt": np.zeros(2)})
        self.assertIsNone(transform)
        self.assertIn("at least 4", diagnostics["reason"])

    def test_frozen_bias_moves_projection_at_the_matching_level(self):
        points = np.array([[0.0, 0.0, 0.025], [0.0, 0.0, 0.003]])
        normals = np.array([[1.0, 0.0], [1.0, 0.0]])
        baseline, _ = fit_rims6.rim_points(points, normals, np.eye(3),
                                            np.array([0.0, 0.0, 1.0]))
        biased, _ = fit_rims6.rim_points(points, normals, np.eye(3),
                                          np.array([0.0, 0.0, 1.0]),
                                          np.array([0, 1]), np.array([0.0002, 0.0]))
        self.assertGreater(biased[0, 0], baseline[0, 0])
        self.assertAlmostEqual(biased[1, 0], baseline[1, 0])

    def test_rig_disagreement_is_measured_at_observed_board_depth(self):
        reference_board = np.eye(4)
        reference_board[2, 3] = 1.0
        camera_board = reference_board.copy()
        reference_camera = np.eye(4)
        reference_camera[:3, :3] = cv2.Rodrigues(
            np.array([0.0, np.deg2rad(1.0), 0.0]))[0]

        error_mm = stationary._fixed_rig_board_error_mm(
            reference_board, camera_board, reference_camera)

        self.assertGreater(error_mm, 10.0)

    def test_weak_or_inconsistent_rig_seed_requests_global_retry(self):
        reference_board = np.eye(4)
        reference_board[2, 3] = 1.0
        reference_row = {"T_cam_board": reference_board.tolist(), "mad_px": 0.2}
        secondary_row = {"T_cam_board": reference_board.tolist(), "mad_px": 0.2}

        self.assertIsNone(stationary._rig_seeded_fit_rejection(
            reference_row, secondary_row, np.eye(4), 4.0))
        secondary_row["mad_px"] = 1.1
        self.assertIn("MAD", stationary._rig_seeded_fit_rejection(
            reference_row, secondary_row, np.eye(4), 4.0))
        secondary_row["mad_px"] = 0.2
        secondary_board = reference_board.copy()
        secondary_board[0, 3] += 0.010
        secondary_row["T_cam_board"] = secondary_board.tolist()
        self.assertIn("disagrees", stationary._rig_seeded_fit_rejection(
            reference_row, secondary_row, np.eye(4), 4.0))

    def test_fast_recalibration_requires_complete_stable_leave_one_out(self):
        diagnostics = {
            "local_transform_diagnostics": {
                "median_corner_error_mm": 1.5,
                "capture_pose_diversity": {
                    "flat_board_yaw_span_deg": 50.0,
                    "flat_board_center_span_mm": 120.0,
                },
            },
            "leave_one_out_validation": {
                "status": "completed",
                "folds": [{"status": "accepted"}] * 5,
                "worst_influence": {"board_corner_max_mm": 0.9,
                                    "rotation_deg": 0.09},
            },
        }
        self.assertIsNone(rig._fast_recalibration_gates(diagnostics))
        diagnostics["leave_one_out_validation"]["folds"][3] = {"status": "unavailable"}
        self.assertIn("not every", rig._fast_recalibration_gates(diagnostics))
        diagnostics["leave_one_out_validation"]["folds"][3] = {"status": "accepted"}
        diagnostics["leave_one_out_validation"]["worst_influence"]["rotation_deg"] = 0.11
        self.assertIn("rotation", rig._fast_recalibration_gates(diagnostics))

    def test_fast_recalibration_requires_flat_board_pose_diversity(self):
        diagnostics = {
            "local_transform_diagnostics": {
                "median_corner_error_mm": 1.5,
                "capture_pose_diversity": {
                    "flat_board_yaw_span_deg": 30.0,
                    "flat_board_center_span_mm": 120.0,
                },
            },
            "leave_one_out_validation": {
                "status": "completed",
                "folds": [{"status": "accepted"}] * 5,
                "worst_influence": {"board_corner_max_mm": 0.9,
                                    "rotation_deg": 0.09},
            },
        }
        self.assertIn("yaw span", rig._fast_recalibration_gates(diagnostics))
        diagnostics["local_transform_diagnostics"]["capture_pose_diversity"][
            "flat_board_yaw_span_deg"] = 50.0
        diagnostics["local_transform_diagnostics"]["capture_pose_diversity"][
            "flat_board_center_span_mm"] = 80.0
        self.assertIn("centre movement", rig._fast_recalibration_gates(diagnostics))


if __name__ == "__main__":
    unittest.main()
