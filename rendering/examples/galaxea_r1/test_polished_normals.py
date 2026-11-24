"""Crease-local shading normals, material seams and normal-only glTF output."""
import base64
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from _panda_scene import write_mesh
from galaxea_r1.polished_normals import _smooth, smooth_painted_shells
from phymeo_rendering.assets import Material
from phymeo_rendering.mesh_data import _local_path, load_mesh
from phymeo_rendering.scene import Instance, Pose, VisualScene


def _triangle(corners, uv_offset=0.):
    p = np.array(corners, dtype="<f4")
    n = np.cross(p[1].astype(float)-p[0], p[2].astype(float)-p[0])
    normal = n / np.linalg.norm(n)
    return SimpleNamespace(positions=p, normals=np.tile(normal, (3, 1)).astype("<f4"),
                           triangles=np.array([[0,1,2]]), texcoords=p[:, :2]/4 + .5 + uv_offset)


def _non_normal_bytes(mesh):
    document = json.loads(_local_path(mesh.uri).read_bytes())
    buffer = document["buffers"][0]
    data = base64.b64decode(buffer.pop("uri").split(",")[1])
    index = document["meshes"][0]["primitives"][0]["attributes"]["NORMAL"]
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    start = view["byteOffset"]
    return document, data[:start], data[start+view["byteLength"]:]


class PaintedShellNormals(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.c = (0,0,0), (0,1,0), (-1,0,0)
        self.d = (1,0,.22)
        self.left = _triangle([self.a,self.b,self.c])
        self.right = _triangle([self.b,self.a,self.d])
        self.crease = _triangle([self.a,self.c,(0,0,1)])

    def test_open_curve_smooths_at_hard_edge_from_its_own_side(self):
        changed, row = _smooth([self.left, self.right, self.crease])
        self.assertGreater(row["changed_vertices"], 0)
        self.assertLess(row["polished_admitted_edge_jump_rms_deg"], row["source_admitted_edge_jump_rms_deg"])
        np.testing.assert_array_equal(changed[0][:2], changed[1][[1,0]])
        self.assertGreater(np.linalg.norm(changed[0][0] - self.left.normals[0]), .01)
        np.testing.assert_array_equal(changed[2], self.crease.normals)

    def test_discontinuous_uv_nearby_layer_and_nonmanifold_edge(self):
        split = _triangle([self.b,self.a,self.d], uv_offset=.125)
        nearby = _triangle(np.array([self.b,self.a,self.d]) + [0,0,.000001])
        point = _triangle([self.a,(0,-1,0),(1,-1,.1)])
        extra = _triangle([self.a,self.b,(-1,0,.3)])
        for chunks in ([self.left,split], [self.left,nearby], [self.left,point], [self.left,self.right,extra]):
            with self.subTest(count=len(chunks)):
                changed, row = _smooth(chunks)
                self.assertEqual(row["changed_vertices"], 0)
                for mesh, normals in zip(chunks, changed):
                    np.testing.assert_array_equal(normals, mesh.normals)

    def test_registered_chunks_and_shared_left_right_meshes(self):
        artifacts = Path(__file__).resolve().parents[3]/"artifacts"
        artifacts.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=artifacts) as tmp:
            root = Path(tmp)
            meshes = tuple(write_mesh(root, f"part-{i}", m.positions, m.normals, m.triangles, texcoords=m.texcoords)
                           for i,m in enumerate([self.left,self.right]))
            sides = ("left_gripper_link", "right_gripper_link")
            instances = tuple(Instance(1+2*s+i, mesh.id, "paint", link, Pose())
                              for s,link in enumerate(sides) for i,mesh in enumerate(meshes))
            scene = VisualScene("painted-test", meshes, (Material("paint", (.8,.8,.8), 0., .24),), instances)
            result = smooth_painted_shells(scene, root/"polished", targets=sides)
            receipt = (root/"polished"/"polished-normals.json").read_bytes()
            report = json.loads(receipt)
            self.assertEqual(result.key, scene.key+":polished-normals-"+sha256(receipt).hexdigest()[:12])
            self.assertEqual(report["source_scene"], scene.key)
            self.assertEqual(report["targets"], list(sides))
            self.assertEqual(len(report["groups"]), 1)
            self.assertEqual(report["groups"][0]["links"], list(sides))
            self.assertEqual((result.instances,result.materials), (scene.instances,scene.materials))
            for src, dst, item in zip(meshes, result.meshes, report["groups"][0]["meshes"]):
                self.assertEqual(src.id, dst.id)
                self.assertEqual(_non_normal_bytes(src), _non_normal_bytes(dst))
                a, b = load_mesh(src), load_mesh(dst)
                self.assertEqual((a.positions,a.texcoords,a.triangles), (b.positions,b.texcoords,b.triangles))
                self.assertNotEqual(a.normals, b.normals)
                self.assertEqual(item["source_sha256"], a.source_sha256)
                self.assertEqual(item["derived_sha256"], b.source_sha256)
                self.assertGreater(item["changed_normals"], 0)
            partial = smooth_painted_shells(scene, root/"one-side", targets=sides[:1])
            self.assertEqual(partial.meshes, scene.meshes)


if __name__ == "__main__":
    unittest.main()
