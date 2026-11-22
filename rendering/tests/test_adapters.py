"""G2 authored value sources: eager snapshots, independent iteration and timelines."""

import math
import pickle
import unittest
from dataclasses import replace

from phymeo_rendering.adapters import AuthoredSceneSource
from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.scene import (
    FrameKey,
    Instance,
    NodePose,
    Pose,
    SceneFrame,
    VisualScene,
)


def _scene():
    return VisualScene(
        "table",
        (Mesh("box", "not-loaded.glb"),),
        (Material("paint", (0.2, 0.3, 0.4), 0, 0.5),),
        (
            Instance(1, "box", "paint", "tool", Pose((0.1, 0, 0))),
            Instance(2, "box", "paint", "object"),
        ),
    )


def _nodes(x=0.0):
    return (
        NodePose("tool", Pose((x, 0.0, 0.0))),
        NodePose("object", Pose((0.0, 1.0, 0.0))),
    )


def _frame(
    *,
    world="A",
    epoch=0,
    revision=2,
    tick=3,
    time=0.125,
    phase="authored",
    scene_key="table",
    nodes=None,
):
    return SceneFrame(
        FrameKey(scene_key, world, epoch, revision, tick, phase),
        time,
        _nodes() if nodes is None else nodes,
    )


class AuthoredSceneSourceTests(unittest.TestCase):
    def test_scene_and_frames_preserve_declared_values(self):
        scene = _scene()
        first = _frame()
        second = _frame(revision=3, tick=8, time=0.456, nodes=_nodes(0.25))
        before = pickle.dumps((scene, first, second))
        source = AuthoredSceneSource(scene, (first, second))
        self.assertEqual(source.scene(), scene)
        self.assertEqual(tuple(source.frames()), (first, second))
        self.assertEqual(pickle.dumps((scene, first, second)), before)
        self.assertEqual(len(source.scene().meshes), 1)
        self.assertEqual(len(source.scene().instances), 2)
        # The adapter must not bake the visual local offset into the node pose.
        self.assertEqual(next(source.frames()).nodes[0].world_pose.position_m, (0.0, 0.0, 0.0))

    def test_finite_generator_is_consumed_once_before_publication(self):
        frames = (_frame(), _frame(revision=3, tick=4, time=0.25))
        consumed = []

        def input_frames():
            for frame in frames:
                consumed.append(frame.key)
                yield frame

        source = AuthoredSceneSource(_scene(), input_frames())
        self.assertEqual(consumed, [frame.key for frame in frames])
        self.assertEqual(tuple(source.frames()), frames)
        self.assertEqual(tuple(source.frames()), frames)
        self.assertEqual(len(consumed), 2)

    def test_iterators_are_independent_and_restart_at_first_frame(self):
        first = _frame()
        second = _frame(revision=3, tick=4, time=0.25)
        source = AuthoredSceneSource(_scene(), (first, second))
        left, right = source.frames(), source.frames()
        self.assertEqual(next(left), first)
        self.assertEqual(next(left), second)
        with self.assertRaises(StopIteration):
            next(left)
        self.assertEqual(next(right), first)
        self.assertEqual(next(right), second)
        self.assertEqual(next(source.frames()), first)

    def test_later_mutation_of_input_sequence_does_not_change_snapshot(self):
        scene = _scene()
        first = _frame()
        second = _frame(revision=3, tick=4, time=0.25)
        supplied = [first, second]
        source = AuthoredSceneSource(scene, supplied)
        before = pickle.dumps((source.scene(), tuple(source.frames())))
        supplied[:] = [_frame(world="unrelated", nodes=_nodes(99.0))]
        supplied.clear()
        self.assertEqual(pickle.dumps((source.scene(), tuple(source.frames()))), before)
        self.assertEqual(tuple(source.frames()), (first, second))

    def test_empty_finite_sequence_is_legal(self):
        scene = _scene()
        source = AuthoredSceneSource(scene, ())
        self.assertEqual(source.scene(), scene)
        self.assertEqual(tuple(source.frames()), ())
        self.assertEqual(tuple(source.frames()), ())

    def test_wrong_scene_or_iterable_type_is_rejected(self):
        with self.assertRaises(TypeError):
            AuthoredSceneSource(object(), ())
        with self.assertRaises(TypeError):
            AuthoredSceneSource(_scene(), None)
        for value in (None, object(), {"key": "table"}, "frame"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                AuthoredSceneSource(_scene(), (_frame(), value))

    def test_invalid_tail_is_rejected_during_construction(self):
        with self.assertRaisesRegex(ValueError, "scene_key"):
            AuthoredSceneSource(_scene(), iter((_frame(), _frame(scene_key="wrong"))))

    def test_input_iterator_error_is_not_hidden_as_a_partial_source(self):
        def failing_frames():
            yield _frame()
            raise RuntimeError("input iteration failed")

        with self.assertRaisesRegex(RuntimeError, "input iteration failed"):
            AuthoredSceneSource(_scene(), failing_frames())

    def test_missing_instance_node_uses_scene_contract(self):
        frame = _frame(nodes=(NodePose("tool", Pose()),))
        with self.assertRaisesRegex(ValueError, "object"):
            AuthoredSceneSource(_scene(), (frame,))

    def test_extra_mount_node_is_preserved(self):
        frame = _frame(nodes=(*_nodes(), NodePose("wrist-camera", Pose((0, 0, 1)))))
        source = AuthoredSceneSource(_scene(), (frame,))
        self.assertEqual(tuple(source.frames()), (frame,))

    def test_authored_source_rejects_other_valid_g1_phases(self):
        for phase in ("step_input", "accepted_endpoint"):
            with self.subTest(phase=phase):
                with self.assertRaisesRegex(ValueError, "authored"):
                    AuthoredSceneSource(_scene(), (_frame(phase=phase),))

    def test_repeated_equal_complete_key_is_allowed_without_deduplication(self):
        first = _frame()
        repeated = replace(first)
        source = AuthoredSceneSource(_scene(), (first, repeated))
        self.assertEqual(tuple(source.frames()), (first, repeated))

    def test_repeated_key_compares_nodes_by_identity_not_tuple_order(self):
        first = _frame()
        repeated = replace(first, nodes=tuple(reversed(first.nodes)))
        source = AuthoredSceneSource(_scene(), (first, repeated))
        self.assertEqual(tuple(source.frames()), (first, repeated))

    def test_quaternion_double_cover_is_same_pose_without_rewriting_input(self):
        half = math.sqrt(0.5)
        first = _frame(
            nodes=(
                NodePose("tool", Pose((0, 0, 0), (0, 0, half, half))),
                NodePose("object", Pose()),
            )
        )
        repeated = replace(
            first,
            nodes=(
                NodePose("tool", Pose((0, 0, 0), (0, 0, -half, -half))),
                NodePose("object", Pose()),
            ),
        )
        before = pickle.dumps((first, repeated))
        source = AuthoredSceneSource(_scene(), (first, repeated))
        self.assertEqual(pickle.dumps(tuple(source.frames())), before)

    def test_repeated_key_rejects_conflicting_time(self):
        first = _frame()
        with self.assertRaises(ValueError) as caught:
            AuthoredSceneSource(_scene(), (first, replace(first, time_s=0.126)))
        self.assertIn("time_s", str(caught.exception))
        self.assertIn("A", str(caught.exception))

    def test_repeated_key_rejects_conflicting_node_pose_or_membership(self):
        first = _frame()
        changed = (
            replace(first, nodes=_nodes(math.nextafter(0.0, 1.0))),
            replace(
                first,
                nodes=(
                    NodePose("tool", Pose(orientation_xyzw=(0, 0, 1, 0))),
                    NodePose("object", Pose((0, 1, 0))),
                ),
            ),
            replace(first, nodes=(*first.nodes, NodePose("mount", Pose()))),
        )
        for candidate in changed:
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError) as caught:
                    AuthoredSceneSource(_scene(), (first, candidate))
                self.assertIn("A", str(caught.exception))
                self.assertIn("node", str(caught.exception))

    def test_timelines_are_independent_for_interleaved_worlds(self):
        frames = (
            _frame(world="A", tick=10, revision=20, time=10.0, nodes=_nodes(1.0)),
            _frame(world="B", tick=3, revision=2, time=0.1, nodes=_nodes(-1.0)),
            _frame(world="A", tick=12, revision=21, time=10.01, nodes=_nodes(1.5)),
            _frame(world="B", tick=5, revision=3, time=0.2, nodes=_nodes(-1.5)),
        )
        source = AuthoredSceneSource(_scene(), frames)
        self.assertEqual(tuple(source.frames()), frames)

    def test_epoch_increase_resets_only_its_own_world_clock(self):
        frames = (
            _frame(world="A", epoch=4, tick=30, revision=40, time=12.5),
            _frame(world="B", epoch=1, tick=3, revision=5, time=2.5),
            _frame(world="A", epoch=6, tick=0, revision=0, time=0.0),
            _frame(world="B", epoch=1, tick=4, revision=6, time=2.6),
        )
        source = AuthoredSceneSource(_scene(), frames)
        self.assertEqual(tuple(source.frames()), frames)

    def test_same_epoch_rejects_each_regressing_clock_component(self):
        first = _frame(tick=10, revision=20, time=1.0)
        candidates = (
            ("tick", _frame(tick=9, revision=21, time=1.1)),
            ("revision", _frame(tick=11, revision=19, time=1.1)),
            ("time_s", _frame(tick=11, revision=21, time=0.9)),
        )
        for field, candidate in candidates:
            with self.subTest(field=field):
                with self.assertRaises(ValueError) as caught:
                    AuthoredSceneSource(_scene(), (first, candidate))
                self.assertIn(field, str(caught.exception))
                self.assertIn("A", str(caught.exception))

    def test_epoch_rollback_is_rejected_even_with_newer_tick_and_time(self):
        frames = (
            _frame(epoch=2, revision=5, tick=10, time=1.0),
            _frame(epoch=1, revision=6, tick=11, time=1.1),
        )
        with self.assertRaisesRegex(ValueError, "epoch"):
            AuthoredSceneSource(_scene(), frames)

    def test_old_exact_key_cannot_bypass_timeline_after_progress(self):
        first = _frame(tick=10, revision=20, time=1.0)
        second = _frame(tick=11, revision=21, time=1.1)
        with self.assertRaises(ValueError):
            AuthoredSceneSource(_scene(), (first, second, first))

    def test_jump_frames_and_same_tick_new_revision_keep_source_time(self):
        frames = (
            _frame(tick=3, revision=2, time=0.125),
            _frame(tick=300, revision=2, time=0.375),
            _frame(tick=300, revision=3, time=0.376, nodes=_nodes(0.25)),
        )
        source = AuthoredSceneSource(_scene(), frames)
        self.assertEqual(tuple(source.frames()), frames)
        self.assertEqual([frame.time_s for frame in source.frames()], [0.125, 0.375, 0.376])


if __name__ == "__main__":
    unittest.main()
