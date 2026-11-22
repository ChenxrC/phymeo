"""Camera validation and exact exposure binding; no device or projection API.

CALIBRATION_SAMPLES and WRIST_SAMPLES are hand-derived integration inputs,
not a claim that Camera projects points or resolves a wrist attachment.
"""

import unittest
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from math import inf, nan, sqrt

from phymeo_rendering import cameras
from phymeo_rendering.cameras import Camera
from phymeo_rendering.scene import FrameKey, NodePose, Pose, SceneFrame


# (camera-space point in metres, expected pixel centre coordinates, forward depth)
CALIBRATION_SAMPLES = (
    ((0.0, 0.0, -1.0), (320.0, 240.0), 1.0),
    ((0.1, 0.1, -1.0), (370.0, 190.0), 1.0),
    ((-0.639, 0.479, -1.0), (0.5, 0.5), 1.0),
)
# Wrist moves from identity to +90 degrees about world Z at (1, 2, 3).
# A fixed attachment at wrist-local (0.1, 0, 0) gives these camera origins.
WRIST_SAMPLES = (
    (Pose((1.0, 2.0, 3.0)), Pose((1.1, 2.0, 3.0))),
    (
        Pose((1.0, 2.0, 3.0), (0.0, 0.0, sqrt(0.5), sqrt(0.5))),
        Pose((1.0, 2.1, 3.0), (0.0, 0.0, sqrt(0.5), sqrt(0.5))),
    ),
)


def _key():
    return FrameKey("table-v1", "world-a", 2, 11, 7, "accepted_endpoint")


def _camera(**changes):
    values = dict(
        id="front",
        frame_key=_key(),
        world_pose=Pose(),
        width=640,
        height=480,
        fx_px=500.0,
        fy_px=500.0,
        cx_px=320.0,
        cy_px=240.0,
        near_m=0.1,
        far_m=5.0,
    )
    values.update(changes)
    return Camera(**values)


class CameraValidationTests(unittest.TestCase):
    def test_calibrated_exposure_preserves_parameters(self):
        key = _key()
        pose = Pose((1.25, -0.5, 0.75), (0.0, 1.0, 0.0, 0.0))
        camera = _camera(id=" front ", frame_key=key, world_pose=pose)
        self.assertEqual(
            (
                camera.id,
                camera.width,
                camera.height,
                camera.fx_px,
                camera.fy_px,
                camera.cx_px,
                camera.cy_px,
                camera.near_m,
                camera.far_m,
            ),
            (" front ", 640, 480, 500.0, 500.0, 320.0, 240.0, 0.1, 5.0),
        )
        # Do not flip the camera axes, reorder XYZW, or rescale metres.
        self.assertEqual(camera.world_pose.position_m, (1.25, -0.5, 0.75))
        self.assertEqual(camera.world_pose.orientation_xyzw, (0.0, 1.0, 0.0, 0.0))
        self.assertIs(camera.frame_key, key)
        self.assertIs(camera.world_pose, pose)

    def test_off_image_principal_point_is_valid(self):
        camera = _camera(cx_px=-20, cy_px=530)
        self.assertEqual((camera.cx_px, camera.cy_px), (-20, 530))

    def test_real_calibration_values_are_not_coerced(self):
        focal = Fraction(1001, 2)
        camera = _camera(fx_px=focal, fy_px=500, cx_px=0, cy_px=0)
        self.assertIs(camera.fx_px, focal)
        self.assertEqual((camera.fy_px, camera.cx_px, camera.cy_px), (500, 0, 0))

    def test_camera_remains_immutable(self):
        camera = _camera()
        with self.assertRaises(FrozenInstanceError):
            camera.width = 1

    def test_camera_rejects_non_string_id(self):
        for value in (None, 1, True, b"front"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "id"):
                    _camera(id=value)

    def test_camera_rejects_blank_id(self):
        for value in ("", " ", "\t\n"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "id"):
                    _camera(id=value)

    def test_camera_requires_real_source_types(self):
        for field, values in (
            ("frame_key", (None, 7, {"tick": 7})),
            ("world_pose", (None, (0, 0, 0), {"position_m": (0, 0, 0)})),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(TypeError, field):
                        _camera(**{field: value})

    def test_camera_rejects_non_integer_dimensions(self):
        for field in ("width", "height"):
            for value in (True, False, 640.0, "640", None, Fraction(640, 1)):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(TypeError, field):
                        _camera(**{field: value})

    def test_camera_rejects_nonpositive_dimensions(self):
        for field in ("width", "height"):
            for value in (0, -1):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        _camera(**{field: value})

    def test_camera_rejects_non_real_calibration(self):
        for field in ("fx_px", "fy_px", "cx_px", "cy_px", "near_m", "far_m"):
            for value in (True, False, "1", None, 1 + 0j):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(TypeError, field):
                        _camera(**{field: value})

    def test_camera_rejects_nonfinite_calibration(self):
        for field in ("fx_px", "fy_px", "cx_px", "cy_px", "near_m", "far_m"):
            for value in (nan, inf, -inf):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        _camera(**{field: value})

    def test_camera_rejects_unrepresentable_calibration(self):
        values = (10**1000, Fraction(10**1000, 3))
        for field in ("fx_px", "fy_px", "cx_px", "cy_px", "near_m", "far_m"):
            for value in values:
                with self.subTest(field=field, numeric_type=type(value).__name__):
                    with self.assertRaisesRegex(ValueError, field):
                        _camera(**{field: value})

    def test_camera_rejects_nonpositive_focal_lengths(self):
        for field in ("fx_px", "fy_px"):
            for value in (0, -0.0, -1):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        _camera(**{field: value})

    def test_camera_rejects_invalid_clip_range(self):
        cases = (
            ({"near_m": 0.0}, "near_m"),
            ({"near_m": -0.1}, "near_m"),
            ({"far_m": 0.0}, "far_m"),
            ({"far_m": -1.0}, "far_m"),
            ({"near_m": 0.5, "far_m": 0.5}, "far_m"),
            ({"near_m": 5.0, "far_m": 0.1}, "far_m"),
        )
        for changes, field in cases:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, field):
                    _camera(**changes)


class ExposureBindingTests(unittest.TestCase):
    def test_exposure_accepts_equal_complete_frame_key(self):
        camera = _camera()
        frame = SceneFrame(_key(), 0.007, ())
        self.assertIsNot(camera.frame_key, frame.key)
        self.assertEqual(camera.frame_key, frame.key)
        self.assertIsNone(cameras.validate_exposure(frame, camera))

    def test_exposure_rejects_each_frame_key_mismatch(self):
        camera = _camera()
        changes = dict(
            scene_key="different-scene",
            world_id="world-b",
            epoch=3,
            revision=12,
            tick=8,
            phase="step_input",
        )
        for field, value in changes.items():
            frame = SceneFrame(replace(camera.frame_key, **{field: value}), 0.007, ())
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    cameras.validate_exposure(frame, camera)

    def test_exposure_requires_frame_and_camera_types(self):
        camera = _camera()
        frame = SceneFrame(_key(), 0.007, ())
        for wrong in (None, {"key": frame.key}, frame.key):
            with self.subTest(wrong_frame=wrong):
                with self.assertRaisesRegex(TypeError, "frame"):
                    cameras.validate_exposure(wrong, camera)
        for wrong in (None, {"frame_key": camera.frame_key}, frame):
            with self.subTest(wrong_camera=wrong):
                with self.assertRaisesRegex(TypeError, "camera"):
                    cameras.validate_exposure(frame, wrong)

    def test_binding_has_no_independent_camera_clock_or_state_changes(self):
        pose = Pose((0.0, 0.0, 0.25))
        frame = SceneFrame(_key(), 0.00725, (NodePose("wrist", pose),))
        camera = _camera(world_pose=pose)
        before_frame, before_camera = replace(frame), replace(camera)
        self.assertIsNone(cameras.validate_exposure(frame, camera))
        self.assertEqual(frame, before_frame)
        self.assertEqual(camera, before_camera)
        self.assertEqual(frame.time_s, 0.00725)
        self.assertFalse(hasattr(camera, "time_s"))

    def test_authored_and_input_exposures_use_their_actual_phase(self):
        for phase in ("authored", "step_input", "accepted_endpoint"):
            with self.subTest(phase=phase):
                key = replace(_key(), phase=phase)
                frame = SceneFrame(key, 0.007, ())
                self.assertIsNone(
                    cameras.validate_exposure(frame, _camera(frame_key=replace(key)))
                )

    def test_world_reset_does_not_relabel_old_exposure(self):
        old_camera = _camera()
        reset_key = replace(_key(), epoch=3, revision=0, tick=0)
        reset_frame = SceneFrame(reset_key, 0.0, ())
        with self.assertRaisesRegex(ValueError, "epoch"):
            cameras.validate_exposure(reset_frame, old_camera)
        reset_camera = _camera(frame_key=reset_key)
        self.assertIsNone(cameras.validate_exposure(reset_frame, reset_camera))
        self.assertEqual(old_camera.frame_key, _key())

    def test_two_pre_resolved_wrist_samples_require_their_own_frames(self):
        exposures = []
        for index, (wrist_pose, camera_pose) in enumerate(WRIST_SAMPLES):
            key = replace(_key(), revision=11 + index, tick=7 + index)
            frame = SceneFrame(key, 0.007 + index * 0.001, (NodePose("wrist", wrist_pose),))
            camera = _camera(id="wrist", frame_key=key, world_pose=camera_pose)
            self.assertIsNone(cameras.validate_exposure(frame, camera))
            self.assertEqual(camera.world_pose, camera_pose)
            exposures.append((frame, camera))
        self.assertEqual(exposures[0][1].world_pose.position_m, (1.1, 2.0, 3.0))
        self.assertEqual(exposures[1][1].world_pose.position_m, (1.0, 2.1, 3.0))
        with self.assertRaisesRegex(ValueError, "revision"):
            cameras.validate_exposure(exposures[1][0], exposures[0][1])


if __name__ == "__main__":
    unittest.main()
