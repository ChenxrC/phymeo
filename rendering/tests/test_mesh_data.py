"""Bounded local glTF decoding with independently authored byte fixtures."""

import base64
import hashlib
import json
import math
import struct
import tempfile
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from phymeo_rendering import mesh_data
from phymeo_rendering.assets import Mesh
from phymeo_rendering.mesh_data import TriangleMesh, load_mesh


POSITIONS = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
NORMALS = ((0.0, 0.0, 1.0),) * 3


def _document(*, index_component=5123, interleaved=False):
    if interleaved:
        # Four leading bytes, two VEC3 attributes in each 28-byte vertex record.
        raw = b"HEAD" + b"".join(
            struct.pack("<6fI", *position, *normal, 0x12345678)
            for position, normal in zip(POSITIONS, NORMALS, strict=True)
        )
        views = [{"buffer": 0, "byteOffset": 4, "byteLength": 84,
                  "byteStride": 28, "target": 34962}]
        position_view, normal_view, normal_offset = 0, 0, 12
    else:
        raw = b"".join(struct.pack("<3f", *p) for p in POSITIONS + NORMALS)
        views = [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
            {"buffer": 0, "byteOffset": 36, "byteLength": 36, "target": 34962},
        ]
        position_view, normal_view, normal_offset = 0, 1, 0
    index_bytes = struct.pack("<3" + {5121: "B", 5123: "H", 5125: "I"}[index_component],
                              0, 1, 2)
    views.append({"buffer": 0, "byteOffset": len(raw),
                  "byteLength": len(index_bytes), "target": 34963})
    raw += index_bytes
    return {
        "asset": {"version": "2.0", "generator": "owned unittest fixture"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4,
        }]}],
        "buffers": [{"byteLength": len(raw), "uri":
                     "data:application/octet-stream;base64,"
                     + base64.b64encode(raw).decode("ascii")}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": position_view, "componentType": 5126, "count": 3,
             "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]},
            {"bufferView": normal_view, "byteOffset": normal_offset,
             "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": len(views) - 1, "componentType": index_component,
             "count": 3, "type": "SCALAR"},
        ],
    }


def _replace_buffer(document, raw):
    document["buffers"][0]["byteLength"] = len(raw)
    document["buffers"][0]["uri"] = (
        "data:application/octet-stream;base64," + base64.b64encode(raw).decode("ascii")
    )


class MeshDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "triangle % fixture.gltf"

    def _write(self, document):
        raw = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.path.write_bytes(raw)
        return raw

    def _load(self, document):
        self._write(document)
        return load_mesh(Mesh("sample", self.path.as_uri()))

    def test_small_triangle_has_immutable_values_and_exact_file_sha(self):
        raw = self._write(_document())
        result = load_mesh(Mesh("sample", self.path.as_uri()))
        self.assertIsInstance(result, TriangleMesh)
        self.assertEqual(result.positions, POSITIONS)
        self.assertEqual(result.normals, NORMALS)
        self.assertEqual(result.triangles, ((0, 1, 2),))
        self.assertEqual(result.source_sha256, hashlib.sha256(raw).hexdigest())
        self.assertIsInstance(result.positions, tuple)
        self.assertIsInstance(result.positions[0], tuple)
        with self.assertRaises(FrozenInstanceError):
            result.positions = ()

    def test_digest_binds_file_metadata_as_well_as_geometry(self):
        original = self._load(_document())
        changed = _document()
        changed["asset"]["generator"] = "different authored source"
        new = self._load(changed)
        self.assertEqual(new.positions, original.positions)
        self.assertEqual(new.triangles, original.triangles)
        self.assertNotEqual(new.source_sha256, original.source_sha256)

    def test_plain_local_path_and_file_uri_identify_same_bytes(self):
        self._write(_document())
        by_path = load_mesh(Mesh("sample", str(self.path)))
        by_uri = load_mesh(Mesh("sample", self.path.as_uri()))
        self.assertEqual(by_path, by_uri)

    def test_interleaved_stride_and_accessor_offsets(self):
        result = self._load(_document(interleaved=True))
        self.assertEqual(result.positions, POSITIONS)
        self.assertEqual(result.normals, NORMALS)
        self.assertEqual(result.triangles, ((0, 1, 2),))

    def test_unsigned_index_components(self):
        for component in (5121, 5123, 5125):
            with self.subTest(component=component):
                self.assertEqual(self._load(_document(index_component=component)).triangles,
                                 ((0, 1, 2),))

    def test_explicit_identity_transform_and_default_triangle_mode(self):
        document = _document()
        document["nodes"][0].update(
            translation=[0, 0, 0], rotation=[0, 0, 0, -1], scale=[1, 1, 1]
        )
        del document["meshes"][0]["primitives"][0]["mode"]
        self.assertEqual(self._load(document).positions, POSITIONS)
        document["nodes"][0] = {"mesh": 0, "matrix": [
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
        ]}
        self.assertEqual(self._load(document).positions, POSITIONS)

    def test_real_authored_table_and_block_keep_metre_bounds(self):
        fixtures = Path(__file__).resolve().parent / "fixtures" / "g1"
        for name, vertex_count, triangle_count, extent in (
            ("table", 4, 2, (0.75, 0.5, 0.0)),
            ("block", 24, 12, (0.0625, 0.0625, 0.0625)),
        ):
            with self.subTest(mesh=name):
                path = fixtures / f"{name}.gltf"
                result = load_mesh(Mesh(name, path.as_uri()))
                self.assertEqual(len(result.positions), vertex_count)
                self.assertEqual(len(result.triangles), triangle_count)
                self.assertEqual(result.source_sha256, hashlib.sha256(path.read_bytes()).hexdigest())
                for axis, half_extent in zip(zip(*result.positions), extent, strict=True):
                    self.assertEqual((min(axis), max(axis)), (-half_extent, half_extent))

    def test_remote_resources_and_uri_parameters_are_rejected(self):
        for uri in (
            "https://example.invalid/mesh.gltf",
            "file://remote-server/share/mesh.gltf",
            "//remote-server/share/mesh.gltf",
            "\\\\remote-server\\share\\mesh.gltf",
            self.path.as_uri() + "?version=1",
            self.path.as_uri() + "#mesh=0",
        ):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                load_mesh(Mesh("sample", uri))

    def test_missing_file_and_glb_have_explicit_errors(self):
        with self.assertRaises(OSError):
            load_mesh(Mesh("sample", self.path.as_uri()))
        with self.assertRaisesRegex(ValueError, "gltf|GLB|glb"):
            load_mesh(Mesh("sample", str(self.path.with_suffix(".glb"))))
        with self.assertRaises(TypeError):
            load_mesh(object())

    def test_file_byte_limit_is_checked_before_json_decode(self):
        self._write(_document())
        with patch.object(mesh_data, "_MAX_FILE_BYTES", 64):
            with patch.object(json, "loads", side_effect=AssertionError("decoded oversize JSON")):
                with self.assertRaises(ValueError):
                    load_mesh(Mesh("sample", self.path.as_uri()))

    def test_malformed_json_duplicate_keys_and_nonfinite_json_rejected(self):
        for raw in (
            b"{", b"[]", b"{\"asset\":{},\"asset\":{}}", b"{\"extras\":NaN}",
            b"{\"extras\":1e999}", b"\xff",
        ):
            self.path.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                load_mesh(Mesh("sample", self.path.as_uri()))

    def test_json_nesting_and_structure_are_bounded(self):
        document = _document()
        nested = None
        for _ in range(70):
            nested = [nested]
        document["extras"] = nested
        with self.assertRaises(ValueError):
            self._load(document)
        document["extras"] = list(range(5000))
        with self.assertRaises(ValueError):
            self._load(document)

    def test_declared_buffer_limit_precedes_base64_decode(self):
        document = _document()
        document["buffers"][0]["byteLength"] = 10**12
        self._write(document)
        with patch.object(base64, "b64decode", side_effect=AssertionError("decoded oversize buffer")):
            with self.assertRaises(ValueError):
                load_mesh(Mesh("sample", self.path.as_uri()))

    def test_vertex_and_triangle_caps_reject_otherwise_valid_input(self):
        self._write(_document())
        with patch.object(mesh_data, "_MAX_VERTICES", 2):
            with self.assertRaises(ValueError):
                load_mesh(Mesh("sample", self.path.as_uri()))
        with patch.object(mesh_data, "_MAX_TRIANGLES", 0):
            with self.assertRaises(ValueError):
                load_mesh(Mesh("sample", self.path.as_uri()))

    def test_external_buffer_and_invalid_embedded_data_are_rejected(self):
        for uri in ("buffer.bin", "https://example.invalid/buffer.bin", "data:,abc"):
            document = _document()
            document["buffers"][0]["uri"] = uri
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                self._load(document)
        document = _document()
        uri = document["buffers"][0]["uri"]
        document["buffers"][0]["uri"] = uri.replace(",", ",!", 1)[:-1]
        with self.assertRaises(ValueError):
            self._load(document)
        document = _document()
        document["buffers"][0]["byteLength"] -= 1
        with self.assertRaises(ValueError):
            self._load(document)

    def test_transforms_and_multiple_scene_objects_are_not_ignored(self):
        node_changes = (
            {"translation": [1, 0, 0]}, {"scale": [2, 2, 2]},
            {"rotation": [0, 0, 1, 0]}, {"children": [0]},
            {"matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1]},
        )
        for fields in node_changes:
            document = _document()
            document["nodes"][0].update(fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self._load(document)
        for collection in ("nodes", "meshes", "scenes"):
            document = _document()
            document[collection].append(deepcopy(document[collection][0]))
            with self.subTest(collection=collection), self.assertRaises(ValueError):
                self._load(document)

    def test_identity_matrix_cannot_be_mixed_with_trs(self):
        document = _document()
        document["nodes"][0].update(
            matrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            translation=[0, 0, 0],
        )
        with self.assertRaises(ValueError):
            self._load(document)

    def test_materials_modes_extensions_and_extra_attributes_are_not_ignored(self):
        for field, value in (
            ("materials", [{}]), ("textures", [{}]), ("animations", [{}]),
            ("extensionsUsed", ["KHR_materials_unlit"]),
            ("extensionsRequired", ["KHR_draco_mesh_compression"]),
        ):
            document = _document()
            document[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self._load(document)
        for field, value in (("material", 0), ("mode", 1), ("targets", [])):
            document = _document()
            document["meshes"][0]["primitives"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self._load(document)
        document = _document()
        document["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"] = 0
        with self.assertRaises(ValueError):
            self._load(document)

    def test_missing_normals_multiple_primitives_and_wrong_version_rejected(self):
        document = _document()
        del document["meshes"][0]["primitives"][0]["attributes"]["NORMAL"]
        with self.assertRaises(ValueError):
            self._load(document)
        document = _document()
        primitives = document["meshes"][0]["primitives"]
        primitives.append(deepcopy(primitives[0]))
        with self.assertRaises(ValueError):
            self._load(document)
        document = _document()
        document["asset"]["version"] = "1.0"
        with self.assertRaises(ValueError):
            self._load(document)

    def test_accessor_counts_types_and_sparse_data_are_validated(self):
        changes = (
            (0, "count", 10**12), (0, "count", True),
            (0, "type", "VEC4"), (0, "componentType", 5123),
            (0, "normalized", True), (0, "sparse", {}),
            (1, "count", 2), (2, "componentType", 5126),
            (2, "count", 2), (2, "byteOffset", -2),
        )
        for index, field, value in changes:
            document = _document()
            document["accessors"][index][field] = value
            with self.subTest(index=index, field=field), self.assertRaises(ValueError):
                self._load(document)

    def test_view_ranges_alignment_stride_and_targets_are_validated(self):
        changes = (
            (0, "byteOffset", -1), (0, "byteOffset", 1),
            (0, "byteLength", 35), (0, "byteLength", 10**12),
            (0, "byteStride", 8), (0, "byteStride", 14),
            (0, "buffer", 1), (0, "target", 34963),
            (2, "byteStride", 4),
        )
        for index, field, value in changes:
            document = _document()
            document["bufferViews"][index][field] = value
            with self.subTest(index=index, field=field), self.assertRaises(ValueError):
                self._load(document)
        document = _document()
        document["accessors"][0]["byteOffset"] = 4
        with self.assertRaises(ValueError):
            self._load(document)

    def test_bad_float_payload_and_degenerate_geometry_are_rejected(self):
        for offset, value in ((0, math.nan), (4, math.inf), (44, 0.0), (44, 2.0)):
            document = _document()
            raw = bytearray(base64.b64decode(document["buffers"][0]["uri"].split(",", 1)[1]))
            struct.pack_into("<f", raw, offset, value)
            _replace_buffer(document, bytes(raw))
            with self.subTest(offset=offset, value=value), self.assertRaises(ValueError):
                self._load(document)
        for indices in ((0, 1, 3), (0, 1, 1)):
            document = _document()
            raw = bytearray(base64.b64decode(document["buffers"][0]["uri"].split(",", 1)[1]))
            struct.pack_into("<3H", raw, 72, *indices)
            _replace_buffer(document, bytes(raw))
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self._load(document)

    def test_errors_identify_asset_and_problem_field(self):
        document = _document()
        document["accessors"][0]["count"] = 0
        with self.assertRaises(ValueError) as caught:
            self._load(document)
        self.assertIn("sample", str(caught.exception))
        self.assertIn("count", str(caught.exception))

    def test_triangle_mesh_constructor_rejects_mutable_or_invalid_values(self):
        digest = "0" * 64
        cases = (
            (list(POSITIONS), NORMALS, ((0, 1, 2),), digest),
            (POSITIONS, list(NORMALS), ((0, 1, 2),), digest),
            (POSITIONS, NORMALS, ([0, 1, 2],), digest),
            (POSITIONS, NORMALS, ((False, 1, 2),), digest),
            (POSITIONS, NORMALS, ((0, 1, 3),), digest),
            (POSITIONS, NORMALS, ((0, 1, 2),), "not-a-digest"),
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                TriangleMesh(*values)


if __name__ == "__main__":
    unittest.main()
