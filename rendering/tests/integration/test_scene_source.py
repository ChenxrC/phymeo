import base64
import hashlib
import json
import struct
import unittest
from dataclasses import replace

from phymeo_rendering.adapters import AuthoredSceneSource
from phymeo_rendering.scene import Pose, compose_pose, validate_scene_frame

from ._g1_table import ALUMINIUM_TOP, FIXTURES, table_frames, table_scene


class TableSceneTest(unittest.TestCase):
    def test_pinned_geometry_and_metre_bounds(self):
        fixtures = (
            (
                "table",
                (0.75, 0.5, 0.0),
                "333f2f88f1522b7a330f3c9e280f7af7efb620407623a780624b792add20ac90",
            ),
            (
                "block",
                (0.0625,) * 3,
                "f7b79c11a537a9ddd85b160cee5fb75c6442739be3dbe5164318dc4079b55713",
            ),
        )
        for name, half_extent, sha256 in fixtures:
            with self.subTest(mesh=name):
                asset = json.loads((FIXTURES / f"{name}.gltf").read_text(encoding="utf-8"))
                buffer = asset["buffers"][0]
                raw = base64.b64decode(buffer["uri"].split(",", 1)[1], validate=True)
                self.assertEqual(len(raw), buffer["byteLength"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), sha256)
                attrs = asset["meshes"][0]["primitives"][0]["attributes"]
                accessor = asset["accessors"][attrs["POSITION"]]
                view = asset["bufferViews"][accessor["bufferView"]]
                start = view["byteOffset"]
                positions = tuple(
                    struct.iter_unpack("<3f", raw[start : start + view["byteLength"]])
                )
                self.assertEqual(len(positions), accessor["count"])
                for axis, extent in zip(zip(*positions), half_extent, strict=True):
                    self.assertEqual((min(axis), max(axis)), (-extent, extent))

    def test_two_published_frames_share_the_same_material_samples(self):
        frames = table_frames()
        source = AuthoredSceneSource(table_scene(), iter(frames))
        scene = source.scene()
        self.assertEqual(
            [m.uri for m in scene.meshes],
            [
                (FIXTURES / "table.gltf").as_uri(),
                (FIXTURES / "block.gltf").as_uri(),
            ],
        )
        samples = [item for item in scene.instances if item.id in (11, 12, 13)]
        self.assertEqual([item.mesh_id for item in samples], ["block"] * 3)
        self.assertEqual([item.material_id for item in samples], ["aluminium", "plastic", "rubber"])
        self.assertEqual(tuple(source.frames()), frames)
        for frame in source.frames():
            validate_scene_frame(scene, frame)
            nodes = {node.node_id: node.world_pose for node in frame.nodes}
            block = compose_pose(nodes[samples[0].node_id], samples[0].local_pose)
            marker = compose_pose(block, Pose(ALUMINIUM_TOP))
            self.assertEqual(marker.position_m, (-0.25, 0.0, 0.0625))
            self.assertIn("wrist", nodes)

    def test_broken_mesh_reference_and_removed_node_do_not_publish(self):
        scene = table_scene()
        frame, _ = table_frames()
        with self.assertRaises(ValueError):
            validate_scene_frame(replace(scene, meshes=scene.meshes[:1]), frame)
        with self.assertRaises(ValueError):
            validate_scene_frame(scene, replace(frame, nodes=frame.nodes[1:]))
