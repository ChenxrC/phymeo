"""Independent FLOAT VEC2 byte fixtures; existing mesh admission stays strict."""
import base64
from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

from phymeo_rendering.assets import Mesh
from phymeo_rendering.mesh_data import TriangleMesh, load_mesh

POSITIONS = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
NORMALS = ((0., 0., 1.),) * 3
UVS = ((0., 0.), (1.5, 0.), (-.25, 1.))


def document(*, interleaved=False):
    if interleaved:
        raw = b"".join(struct.pack("<8f", *p, *n, *uv) for p, n, uv in zip(POSITIONS, NORMALS, UVS, strict=True))
        views = [{"buffer": 0, "byteOffset": 0, "byteLength": 96, "byteStride": 32, "target": 34962}]
        views_for = (0, 0, 0)
        offsets = (0, 12, 24)
    else:
        raw = b"".join(struct.pack("<3f", *p) for p in POSITIONS + NORMALS)
        raw += b"".join(struct.pack("<2f", *uv) for uv in UVS)
        views = [{"buffer": 0, "byteOffset": offset, "byteLength": size, "target": 34962}
                 for offset, size in ((0, 36), (36, 36), (72, 24))]
        views_for = (0, 1, 2)
        offsets = (0, 0, 0)
    views.append({"buffer": 0, "byteOffset": len(raw), "byteLength": 6, "target": 34963})
    raw += struct.pack("<3H", 0, 1, 2)
    return {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "mode": 4}]}],
        "buffers": [{"byteLength": len(raw), "uri": "data:application/octet-stream;base64," + base64.b64encode(raw).decode()}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": views_for[0], "byteOffset": offsets[0], "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]},
            {"bufferView": views_for[1], "byteOffset": offsets[1], "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": views_for[2], "byteOffset": offsets[2], "componentType": 5126, "count": 3, "type": "VEC2", "min": [-.25, 0], "max": [1.5, 1]},
            {"bufferView": len(views)-1, "componentType": 5123, "count": 3, "type": "SCALAR"}],
    }


class MeshUVTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "uv.gltf"

    def load(self, obj):
        self.path.write_text(json.dumps(obj), encoding="utf-8")
        return load_mesh(Mesh("uv", self.path.as_uri()))

    def test_optional_uv_is_owned_and_legacy_constructor_is_unchanged(self):
        old = TriangleMesh(POSITIONS, NORMALS, ((0, 1, 2),), "a"*64)
        self.assertIsNone(old.texcoords)
        new = TriangleMesh(POSITIONS, NORMALS, ((0, 1, 2),), "a"*64, UVS)
        self.assertEqual(new.texcoords, UVS)
        with self.assertRaises(FrozenInstanceError):
            new.texcoords = None

    def test_separate_and_interleaved_uv_keep_negative_and_repeated_coordinates(self):
        for interleaved in (False, True):
            with self.subTest(interleaved=interleaved):
                mesh = self.load(document(interleaved=interleaved))
                self.assertEqual((mesh.positions, mesh.normals, mesh.triangles, mesh.texcoords), (POSITIONS, NORMALS, ((0, 1, 2),), UVS))
                self.assertEqual(mesh.source_sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_uv_accessor_type_count_alias_and_bounds_are_enforced(self):
        for field, value in (("type", "VEC3"), ("componentType", 5123), ("count", 2), ("normalized", True),
                             ("min", [0, 0]), ("max", [1.5, 1, 0]), ("byteOffset", 1), ("sparse", {})):
            obj = document()
            obj["accessors"][2][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.load(obj)
        obj = document()
        obj["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"] = 0
        with self.assertRaises(ValueError):
            self.load(obj)

    def test_extra_uv_sets_embedded_materials_and_unused_accessors_still_reject(self):
        for mutate in (
            lambda obj: obj["meshes"][0]["primitives"][0]["attributes"].update(TEXCOORD_1=2),
            lambda obj: obj.update(materials=[{}]),
            lambda obj: obj["accessors"].append(deepcopy(obj["accessors"][2])),
        ):
            obj = document()
            mutate(obj)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.load(obj)

    def test_uv_nonfinite_payload_and_mutable_direct_values_reject(self):
        for value in (math.nan, math.inf, -math.inf):
            obj = document()
            # Bounds are optional, so nonfinite values must be rejected by the
            # owned mesh even when no min/max declaration happens to expose them.
            obj["accessors"][2].pop("min")
            obj["accessors"][2].pop("max")
            raw = bytearray(base64.b64decode(obj["buffers"][0]["uri"].split(",")[1]))
            struct.pack_into("<f", raw, 72, value)
            obj["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(raw).decode()
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(obj)
        for uv in (list(UVS), UVS[:2], ((0.,),)*3, ((0., math.inf),)*3, ([0., 0.],)*3):
            with self.subTest(uv=uv), self.assertRaises((ValueError, TypeError)):
                TriangleMesh(POSITIONS, NORMALS, ((0, 1, 2),), "a"*64, uv)


if __name__ == "__main__":
    unittest.main()
