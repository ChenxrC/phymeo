import unittest
from dataclasses import replace

from phymeo_rendering.cameras import Camera, validate_exposure
from phymeo_rendering.scene import compose_pose, validate_scene_frame

from ._g1_table import CAMERA_MOUNT, table_frames, table_scene


def marker_pixel(camera, world_point):
    """Project using the three world-space axes of this calibrated exposure."""
    x, y, z, w = camera.world_pose.orientation_xyzw
    right = (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w))
    up = (2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w))
    back = (2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y))
    delta = [p - c for p, c in zip(world_point, camera.world_pose.position_m)]
    dx, dy, dz = (sum(a * b for a, b in zip(axis, delta)) for axis in (right, up, back))
    return (camera.cx_px + camera.fx_px * dx / -dz, camera.cy_px - camera.fy_px * dy / -dz, -dz)


class CameraCaptureTest(unittest.TestCase):
    def test_wrist_pose_and_exposure_belong_to_the_same_frame(self):
        scene = table_scene()
        frames = table_frames()
        cameras = []
        for frame, expected in zip(frames, ((195.0, 240.0, 1.0), (320.0, 52.5, 1.0))):
            validate_scene_frame(scene, frame)
            wrist = next(node for node in frame.nodes if node.node_id == "wrist")
            camera = Camera(
                "wrist",
                frame.key,
                compose_pose(wrist.world_pose, CAMERA_MOUNT),
                640,
                480,
                500.0,
                500.0,
                320.0,
                240.0,
                0.1,
                5.0,
            )
            validate_exposure(frame, camera)
            actual = marker_pixel(camera, (-0.25, 0.0, 0.0625))
            for component, reference in zip(actual, expected):
                self.assertAlmostEqual(component, reference, delta=1e-12)
            cameras.append(camera)

        for foreign_key in (
            frames[0].key,
            replace(frames[1].key, epoch=1),
            replace(frames[1].key, world_id="other"),
        ):
            with self.subTest(key=foreign_key), self.assertRaises(ValueError):
                validate_exposure(frames[1], replace(cameras[1], frame_key=foreign_key))
