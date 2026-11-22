"""Real archived Panda states; no simulator or renderer is loaded."""
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from phymeo_rendering.adapters.panda_replay import PandaReplaySource
from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.scene import Instance, Pose, VisualScene

FIXTURE = Path(__file__).parent / "fixtures" / "panda_replay"
EXCERPT_SHA = "cf9f073747d8d5e76a2b9905a27765a9b80db753e65e2a3270bd964b42ea2d98"


def scene():
    return VisualScene("panda-archive-test", (Mesh("unused", "not-loaded.obj"),),
                       (Material("white", (1, 1, 1), 0, .5),), tuple(
                           Instance(i + 1, "unused", "white", name) for i, name in enumerate(
                               ("panda_link0", "panda_hand", "panda_leftfinger",
                                "panda_rightfinger", "cube", "table"))))


def source(root=FIXTURE):
    return PandaReplaySource(scene(), root / "panda.urdf", root / "trace.json")


def point(pose, local):
    # Independent vector quaternion rotation, for recorded input pad witnesses.
    x, y, z, w = pose.orientation_xyzw
    vx, vy, vz = local
    cross = (y*vz-z*vy, z*vx-x*vz, x*vy-y*vx)
    double_cross = (y*cross[2]-z*cross[1], z*cross[0]-x*cross[2], x*cross[1]-y*cross[0])
    return tuple(p + v + 2*w*c + 2*d for p, v, c, d in
                 zip(pose.position_m, local, cross, double_cross))


class PandaReplayTests(unittest.TestCase):
    def test_fixture_exact_identity_and_two_meaningful_states(self):
        self.assertEqual(hashlib.sha256((FIXTURE / "trace.json").read_bytes()).hexdigest(), EXCERPT_SHA)
        value = source()
        frames = list(value.frames())
        self.assertEqual([f.time_s for f in frames], [.3, 1.35])
        self.assertEqual([f.key.tick for f in frames], [480, 2160])
        self.assertEqual([f.key.revision for f in frames], [479, 2159])
        self.assertEqual(frames, list(value.frames()))
        evidence = [value.provenance(f) for f in frames]
        self.assertEqual([e["record_index"] for e in evidence], [479, 2159])
        self.assertIsNone(evidence[0]["live_physics_identity"])
        self.assertEqual(evidence[0]["identity_scope"], "archive-replay")
        self.assertGreater(abs(evidence[0]["record"]["q"][1]-evidence[1]["record"]["q"][1]), .2)
        self.assertEqual(evidence[0]["record"]["pad_normal_N"], [0., 0.])
        self.assertGreater(min(evidence[1]["record"]["pad_normal_N"]), 5.)
        self.assertEqual(evidence[1]["record"]["table_N"], 0.)

    def test_recorded_cube_and_time_are_not_recomputed(self):
        value = source()
        for frame in value.frames():
            raw = value.provenance(frame)["record"]
            nodes = {n.node_id: n.world_pose for n in frame.nodes}
            self.assertEqual(nodes["cube"], Pose(tuple(raw["cube"]), tuple(raw["xyzw"])))
            self.assertEqual(nodes["panda_link0"], Pose())
            self.assertEqual(nodes["table"].position_m, (.3443905665929414, 2.1423834928313568e-16, .39963205230283927))
        geometry = value.provenance(next(value.frames()))["geometry"]
        self.assertEqual(geometry["cube_full_size_m"], [.046, .058, .04])
        self.assertEqual(geometry["table_full_size_m"], [.179, .11, .016])

    def test_urdf_fk_matches_archived_input_pad_points(self):
        value = source()
        for frame in value.frames():
            evidence = value.provenance(frame)
            nodes = {n.node_id: n.world_pose for n in value.link_poses(evidence["record_index"], before=True)}
            raw_points = evidence["record"]["pad_points_before_m"]
            for side, y, offset in (("left", -.00002, 0), ("right", .00002, 4)):
                actual = point(nodes[f"panda_{side}finger"], (0., y, .04525))
                expected = [math.fsum(p[k] for p in raw_points[offset:offset+4])/4 for k in range(3)]
                self.assertLess(math.dist(actual, expected), 2e-12)

    def test_independent_finger_coordinates_are_not_replaced_by_mimic(self):
        value = source()
        frame = list(value.frames())[1]
        q = value.provenance(frame)["record"]["q"]
        nodes = {n.node_id: n.world_pose for n in frame.nodes}
        span = math.dist(nodes["panda_leftfinger"].position_m, nodes["panda_rightfinger"].position_m)
        self.assertAlmostEqual(span, q[7]+q[8], delta=2e-15)
        self.assertNotEqual(q[7], q[8])

    def test_foreign_frame_and_changed_node_are_rejected(self):
        value = source()
        frame = next(value.frames())
        for wrong in (replace(frame, key=replace(frame.key, world_id="live-world")),
                      replace(frame, time_s=frame.time_s+.001),
                      replace(frame, nodes=frame.nodes[:-1])):
            with self.subTest(wrong=wrong.key), self.assertRaises(ValueError):
                value.provenance(wrong)
        with self.assertRaises(ValueError):
            value.link_poses(480)

    def test_wrong_urdf_and_tampered_excerpt_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("panda.urdf", "trace.json", "manifest.json"):
                (root/name).write_bytes((FIXTURE/name).read_bytes())
            (root/"panda.urdf").write_bytes((root/"panda.urdf").read_bytes()+b"\n")
            with self.assertRaises(ValueError):
                source(root)
            (root/"panda.urdf").write_bytes((FIXTURE/"panda.urdf").read_bytes())
            (root/"trace.json").write_bytes((root/"trace.json").read_bytes().replace(b'"source_index":479', b'"source_index":478'))
            with self.assertRaises(ValueError):
                source(root)

    def test_reordered_joints_even_with_rehashed_excerpt_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"panda.urdf").write_bytes((FIXTURE/"panda.urdf").read_bytes())
            raw = (FIXTURE/"trace.json").read_text().replace(
                '"panda_joint1", "panda_joint2"', '"panda_joint2", "panda_joint1"')
            self.assertNotEqual(raw, (FIXTURE/"trace.json").read_text())
            (root/"trace.json").write_text(raw, encoding="utf-8")
            manifest = json.loads((FIXTURE/"manifest.json").read_text())
            manifest["excerpt_sha256"] = hashlib.sha256((root/"trace.json").read_bytes()).hexdigest()
            (root/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "joint"):
                source(root)

    def test_missing_scene_node_and_provenance_mutation(self):
        bad = replace(scene(), instances=(Instance(1, "unused", "white", "foreign_link"),))
        with self.assertRaises(ValueError):
            PandaReplaySource(bad, FIXTURE/"panda.urdf", FIXTURE/"trace.json")
        value = source()
        frame = next(value.frames())
        evidence = value.provenance(frame)
        evidence["record"]["q"][0] = 900
        self.assertNotEqual(value.provenance(frame)["record"]["q"][0], 900)


if __name__ == "__main__":
    unittest.main()
