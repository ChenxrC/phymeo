"""G1 scene contracts; pure standard-library tests, no rendering or physics."""

import math
import pickle
import unittest
from dataclasses import FrozenInstanceError, replace

from phymeo_rendering import scene
from phymeo_rendering.assets import Material, Mesh


def key(**changes):
    fields = dict(
        scene_key="table", world_id="world-0", epoch=0, revision=2,
        tick=3, phase="accepted_endpoint",
    )
    fields.update(changes)
    return scene.FrameKey(**fields)


def table(**changes):
    fields = dict(
        key="table",
        meshes=(Mesh("cube", "cube.obj"),),
        materials=(Material("paint", (0.2, 0.3, 0.4), 0.0, 0.5),),
        instances=(scene.Instance(1, "cube", "paint", "arm"),),
    )
    fields.update(changes)
    return scene.VisualScene(**fields)


class SceneContracts(unittest.TestCase):
    def test_identity_pose_and_empty_authored_scene(self):
        pose = scene.Pose()
        self.assertEqual(pose.position_m, (0.0, 0.0, 0.0))
        self.assertEqual(pose.orientation_xyzw, (0.0, 0.0, 0.0, 1.0))
        empty = scene.VisualScene("empty", (), (), ())
        frame = scene.SceneFrame(key(scene_key="empty", phase="authored"), 0.0, ())
        self.assertIsNone(scene.validate_scene_frame(empty, frame))
        with self.assertRaises(FrozenInstanceError):
            pose.position_m = (1.0, 0.0, 0.0)

    def test_pose_rejects_nonfinite_or_zero_rotation(self):
        invalid = (
            {"position_m": (math.nan, 0, 0)},
            {"position_m": (math.inf, 0, 0)},
            {"position_m": (0, 0)},
            {"orientation_xyzw": (0, 0, 0, 0)},
            {"orientation_xyzw": (0, 0, 0, 2)},
            {"orientation_xyzw": (0, 0, 1)},
            {"orientation_xyzw": (0, 0, -math.inf, 1)},
        )
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                scene.Pose(**fields)

    def test_vectors_reject_mutable_or_nonnumeric_coordinates(self):
        for value in ([0, 0, 0], (True, 0, 0), ("0", 0, 0), ([], 0, 0)):
            with self.subTest(value=value), self.assertRaises(TypeError):
                scene.Pose(position_m=value)
        with self.assertRaises(TypeError):
            scene.Pose(orientation_xyzw=[0, 0, 0, 1])

    def test_unit_tolerance_preserves_supplied_values_and_sign(self):
        for length in (1 - 0.9e-6, 1 + 0.9e-6):
            q = (-0.0, 0.0, 0.0, -length)
            pose = scene.Pose(orientation_xyzw=q)
            self.assertEqual(pickle.dumps(pose.orientation_xyzw), pickle.dumps(q))
            self.assertEqual(scene.DirectionalLight((0, 0, length), (0, 0, 0)).direction,
                             (0, 0, length))
        for length in (1 - 1.1e-6, 1 + 1.1e-6):
            with self.subTest(length=length):
                with self.assertRaises(ValueError):
                    scene.Pose(orientation_xyzw=(0, 0, 0, length))
                with self.assertRaises(ValueError):
                    scene.DirectionalLight((length, 0, 0), (1, 1, 1))

    def test_light_rejects_invalid_direction(self):
        for direction in ((0, 0, 0), (0, 0, 2), (math.nan, 0, 1), (0, 1)):
            with self.subTest(direction=direction), self.assertRaises(ValueError):
                scene.DirectionalLight(direction, (1, 1, 1))
        with self.assertRaises(TypeError):
            scene.DirectionalLight((True, 0, 0), (1, 1, 1))
        for light in ((-1, 0, 0), (0, math.inf, 1), (1, 1)):
            with self.subTest(light=light), self.assertRaises(ValueError):
                scene.DirectionalLight((0, 0, -1), light)
        with self.assertRaises(TypeError):
            scene.DirectionalLight((0, 0, -1), [1, 1, 1])
        self.assertEqual(scene.DirectionalLight((0, 0, -1), (0, 0, 0)).illuminance,
                         (0, 0, 0))

    def test_frame_key_rejects_invalid_identity(self):
        for name in ("scene_key", "world_id"):
            for value in ("", " \t"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    key(**{name: value})
            with self.assertRaises(TypeError):
                key(**{name: 5})
        for name in ("epoch", "revision", "tick"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    key(**{name: -1})
                for value in (False, 1.0, "1"):
                    with self.assertRaises(TypeError):
                        key(**{name: value})
        with self.assertRaises(ValueError):
            key(phase="future")
        with self.assertRaises(TypeError):
            key(phase=None)
        for phase in ("authored", "step_input", "accepted_endpoint"):
            self.assertEqual(key(phase=phase).phase, phase)

    def test_instance_id_bounds_and_named_bindings(self):
        for value in (1, 2**32 - 1):
            self.assertEqual(scene.Instance(value, "cube", "paint", "arm").id, value)
        for value in (0, -1, 2**32):
            with self.assertRaises(ValueError):
                scene.Instance(value, "cube", "paint", "arm")
        for value in (True, 1.0, "1"):
            with self.assertRaises(TypeError):
                scene.Instance(value, "cube", "paint", "arm")
        original = scene.Instance(1, "cube", "paint", "arm")
        for field in ("mesh_id", "material_id", "node_id"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(original, **{field: " "})
        with self.assertRaises(TypeError):
            replace(original, local_pose=())

    def test_scene_rejects_mutable_collections_and_wrong_members(self):
        original = table()
        for field in ("meshes", "materials", "instances", "lights"):
            with self.subTest(field=field):
                with self.assertRaises(TypeError):
                    replace(original, **{field: list(getattr(original, field))})
                with self.assertRaises(TypeError):
                    replace(original, **{field: (object(),)})
        with self.assertRaises(ValueError):
            replace(original, key=" ")
        with self.assertRaises(TypeError):
            replace(original, key=1)

    def test_shared_mesh_with_two_instances_and_extra_mount_node(self):
        original = table(instances=(scene.Instance(1, "cube", "paint", "arm"),
                                    scene.Instance(2, "cube", "paint", "arm")))
        frame = scene.SceneFrame(key(), 0.0123, (
            scene.NodePose("arm", scene.Pose()),
            scene.NodePose("wrist-camera-mount", scene.Pose((1, 2, 3))),
        ))
        before = pickle.dumps((original, frame))
        self.assertIsNone(scene.validate_scene_frame(original, frame))
        self.assertEqual(before, pickle.dumps((original, frame)))

    def test_missing_asset_reference_is_rejected(self):
        for field, missing in (("mesh_id", "missing-mesh"),
                               ("material_id", "missing-material")):
            item = replace(scene.Instance(1, "cube", "paint", "arm"), **{field: missing})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, missing):
                table(instances=(item,))

    def test_duplicate_static_id_is_rejected(self):
        original = table()
        for field in ("meshes", "materials", "instances"):
            entry = getattr(original, field)[0]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, str(entry.id)):
                replace(original, **{field: (entry, entry)})

    def test_duplicate_or_missing_node_is_rejected(self):
        node = scene.NodePose("arm", scene.Pose())
        with self.assertRaisesRegex(ValueError, "arm"):
            scene.SceneFrame(key(), 0.0, (node, node))
        frame = scene.SceneFrame(key(), 0.0, (scene.NodePose("other", scene.Pose()),))
        with self.assertRaisesRegex(ValueError, "arm"):
            scene.validate_scene_frame(table(), frame)

    def test_scene_key_mismatch_is_rejected(self):
        frame = scene.SceneFrame(key(scene_key="another-table"), 0.0,
                                 (scene.NodePose("arm", scene.Pose()),))
        with self.assertRaisesRegex(ValueError, "scene_key"):
            scene.validate_scene_frame(table(), frame)

    def test_node_pose_and_frame_local_types(self):
        with self.assertRaises(ValueError):
            scene.NodePose(" ", scene.Pose())
        with self.assertRaises(TypeError):
            scene.NodePose("arm", (0, 0, 0))
        with self.assertRaises(TypeError):
            scene.SceneFrame("key", 0.0, ())
        for nodes in ([], (object(),)):
            with self.assertRaises(TypeError):
                scene.SceneFrame(key(), 0.0, nodes)
        for time in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                scene.SceneFrame(key(), time, ())
        for time in (False, "0"):
            with self.assertRaises(TypeError):
                scene.SceneFrame(key(), time, ())

    def test_validation_has_no_tick_history_or_clock_reconstruction(self):
        original = table()
        nodes = (scene.NodePose("arm", scene.Pose()),)
        for tick, epoch, time in ((10, 0, 0.123), (0, 1, 0.0), (3, 0, 1.25), (3, 0, 1.25)):
            frame = scene.SceneFrame(key(tick=tick, epoch=epoch), time, nodes)
            self.assertIsNone(scene.validate_scene_frame(original, frame))
            self.assertEqual(frame.time_s, time)

    def test_public_functions_reject_wrong_types(self):
        with self.assertRaises(TypeError):
            scene.validate_scene_frame(object(), scene.SceneFrame(key(), 0.0, ()))
        with self.assertRaises(TypeError):
            scene.validate_scene_frame(table(), object())
        with self.assertRaises(TypeError):
            scene.compose_pose(scene.Pose(), object())
        with self.assertRaises(TypeError):
            scene.compose_pose(object(), scene.Pose())

    def test_compose_pose_rotates_local_translation(self):
        half = math.sqrt(0.5)
        parent = scene.Pose((1, 2, 0), (0, 0, half, half))
        local = scene.Pose((1, 0, 0))
        result = scene.compose_pose(parent, local)
        for value, expected in zip(result.position_m, (1, 3, 0), strict=True):
            self.assertAlmostEqual(value, expected, delta=1e-12)
        for value, expected in zip(result.orientation_xyzw, parent.orientation_xyzw, strict=True):
            self.assertAlmostEqual(value, expected, delta=1e-12)

    def test_composition_order_and_quaternion_double_cover(self):
        half = math.sqrt(0.5)
        parent = scene.Pose(orientation_xyzw=(0, 0, half, half))
        local = scene.Pose(orientation_xyzw=(half, 0, 0, half))
        expected = (0.5, 0.5, 0.5, 0.5)  # world Z then local X, not reversed.
        result = scene.compose_pose(parent, local)
        for value, want in zip(result.orientation_xyzw, expected, strict=True):
            self.assertAlmostEqual(value, want, delta=1e-12)
        reversed_sign = replace(parent, orientation_xyzw=tuple(-x for x in parent.orientation_xyzw))
        alternative = scene.compose_pose(reversed_sign, local)
        difference = min(max(abs(a - sign * b) for a, b in zip(
            result.orientation_xyzw, alternative.orientation_xyzw, strict=True)) for sign in (-1, 1))
        self.assertLessEqual(difference, 1e-12)

    def test_near_unit_chain_has_no_rotation_scale_or_norm_drift(self):
        length = 1 + 0.9e-6
        half = math.sqrt(0.5)
        parent = scene.Pose((1, 2, 0), (0, 0, half * length, half * length))
        local = scene.Pose((1, 0, 0), (0, 0, 0, length))
        result = scene.compose_pose(parent, local)
        for value, expected in zip(result.position_m, (1, 3, 0), strict=True):
            self.assertAlmostEqual(value, expected, delta=1e-12)
        rotation = scene.Pose(orientation_xyzw=parent.orientation_xyzw)
        combined = scene.Pose()
        for _ in range(1000):
            combined = scene.compose_pose(combined, rotation)
            self.assertAlmostEqual(math.hypot(*combined.orientation_xyzw), 1.0, delta=1e-14)
        # 1000 quarter turns: identity modulo quaternion sign.
        self.assertLessEqual(max(abs(x) for x in combined.orientation_xyzw[:3]), 1e-12)
        self.assertAlmostEqual(abs(combined.orientation_xyzw[3]), 1.0, delta=1e-12)

    def test_composition_does_not_mutate_sources(self):
        parent = scene.Pose((-0.0, 2.0, 1.0), (0.0, 0.0, 0.0, -(1 + 0.9e-6)))
        local = scene.Pose((1.0, 0.0, 0.0))
        before = pickle.dumps((parent, local))
        scene.compose_pose(parent, local)
        self.assertEqual(before, pickle.dumps((parent, local)))

    def test_unrepresentable_composed_position_is_rejected(self):
        with self.assertRaises(ValueError):
            scene.compose_pose(scene.Pose((1e308, 0, 0)), scene.Pose((1e308, 0, 0)))


if __name__ == "__main__":
    unittest.main()
