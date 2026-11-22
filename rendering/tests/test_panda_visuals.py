"""Pinned complete Panda visuals, source semantics and tamper rejection."""

import hashlib
import json
import runpy
import shutil
import tempfile
import unittest
from pathlib import Path

from phymeo_rendering.panda_visuals import panda_visual_assets
from phymeo_rendering.mesh_data import load_mesh

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "panda"
TOOL = ROOT / "tools" / "prepare_panda_visuals.py"

DAE = '''<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit meter="1"/><up_axis>Z_UP</up_axis></asset><library_images/>
<library_effects><effect id="e"><profile_COMMON><technique sid="common"><phong>
<emission><color>0 0 0 1</color></emission><ambient><color>0 0 0 1</color></ambient>
<diffuse><color>0.2 0.4 0.6 1</color></diffuse><specular><color>0.25 0.25 0.25 1</color></specular>
<shininess><float>30</float></shininess><transparent opaque="A_ONE"><color>1 1 1 1</color></transparent>
<transparency><float>1</float></transparency><index_of_refraction><float>1</float></index_of_refraction>
</phong></technique></profile_COMMON></effect></library_effects>
<library_materials><material id="m"><instance_effect url="#e"/></material></library_materials>
<library_geometries><geometry id="g"><mesh>
<source id="p"><float_array id="pa" count="9">0 0 0 1 0 0 0 1 0</float_array>
<technique_common><accessor source="#pa" count="3" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<source id="n"><float_array id="na" count="3">0 0 1</float_array>
<technique_common><accessor source="#na" count="1" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
<triangles material="slot" count="1"><input semantic="VERTEX" source="#v" offset="0"/><input semantic="NORMAL" source="#n" offset="1"/><p>0 0 1 0 2 0</p></triangles>
</mesh></geometry></library_geometries>
<library_visual_scenes><visual_scene id="scene"><node id="node"><matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>
<instance_geometry url="#g"><bind_material><technique_common><instance_material symbol="slot" target="#m"/></technique_common></bind_material></instance_geometry>
</node></visual_scene></library_visual_scenes><scene><instance_visual_scene url="#scene"/></scene></COLLADA>'''


class SourcePreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = runpy.run_path(str(TOOL))

    def decode(self, text=DAE):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "owned.dae"
            path.write_text(text, encoding="utf-8")
            return self.tool["decode_dae"](path)

    def test_source_triangle_color_normals_and_phong_are_preserved(self):
        parts = self.decode()
        self.assertEqual(len(parts), 1)
        part = parts[0]
        self.assertEqual(part["positions"], ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        self.assertEqual(part["normals"], ((0.0, 0.0, 1.0),) * 3)
        self.assertEqual(part["triangles"], ((0, 1, 2),))
        self.assertEqual(part["phong"]["diffuse"], (0.2, 0.4, 0.6, 1.0))
        self.assertEqual(part["phong"]["shininess"], 30.0)
        self.assertEqual(self.tool["pbr_material"](part["phong"]),
                         {"base_color": [0.2, 0.4, 0.6], "metallic": 0.0, "roughness": 0.25})

    def test_unsupported_scene_semantics_are_not_silently_ignored(self):
        variants = (
            DAE.replace('meter="1"', 'meter="0.001"'),
            DAE.replace('Z_UP', 'Y_UP'),
            DAE.replace('0 0 0 1</matrix>', '1 0 0 1</matrix>'),
            DAE.replace('<library_images/>', '<library_images><image id="x"><init_from>x.png</init_from></image></library_images>'),
            DAE.replace('<color>0.2 0.4 0.6 1</color>', '<texture texture="x" texcoord="UV"/>'),
            DAE.replace('<float>1</float></transparency>', '<float>0.5</float></transparency>'),
            DAE.replace('<triangles material=', '<polylist material=').replace('</triangles>', '</polylist>'),
        )
        for text in variants:
            with self.subTest(text=text[:90]), self.assertRaises(ValueError):
                self.decode(text)

    def test_bad_indices_normals_and_counts_are_rejected(self):
        for old, new in (
            ('0 0 1 0 2 0</p>', '0 0 1 0 9 0</p>'),
            ('0 0 1</float_array>', '0 0 0</float_array>'),
            ('material="slot" count="1"', 'material="slot" count="2"'),
            ('semantic="NORMAL"', 'semantic="TEXCOORD"'),
        ):
            with self.subTest(new=new), self.assertRaises(ValueError):
                self.decode(DAE.replace(old, new))

    def test_collada_affine_transform_and_inverse_transpose_are_baked(self):
        source = DAE.replace("1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>",
                             "0 -2 0 3 1 0 0 4 0 0 1 5 0 0 0 1</matrix>")
        part = self.decode(source)[0]
        self.assertEqual(part["positions"], ((3., 4., 5.), (3., 5., 5.), (1., 4., 5.)))
        self.assertEqual(part["normals"], ((0., 0., 1.),) * 3)

    def test_source_zero_area_has_explicit_ordinal_receipt_but_quantized_surface_loss_rejects(self):
        source = DAE.replace('material="slot" count="1"', 'material="slot" count="2"')
        source = source.replace("0 0 1 0 2 0</p>", "0 0 1 0 2 0 0 0 0 0 1 0</p>")
        part, report = self.tool["admitted_part"](self.decode(source)[0])
        self.assertEqual(part["triangles"], ((0, 1, 2),))
        self.assertEqual(report["source_triangle_count"], 2)
        self.assertEqual(report["source_triangle_runs"], ((0, 1),))
        self.assertEqual([item["source_ordinal"] for item in report["omitted_zero_area_triangles"]], [1])
        self.assertEqual(report["source_zero_area_count"], 1)
        self.assertEqual(report["float32_zero_area_count"], 1)
        self.assertEqual(report["quantization_surface_loss_count"], 0)
        collapsed = self.decode()[0]
        collapsed["positions"] = ((1., 0., 0.), (1.+1e-9, 0., 0.), (1., 1., 0.))
        collapsed["source_positions"] = collapsed["positions"]
        with self.assertRaisesRegex(ValueError, "positive-area.*FLOAT32"):
            self.tool["admitted_part"](collapsed)

    def test_normal_uses_inverse_transpose_and_zero_surface_part_is_explicit_rejection(self):
        import math
        source = DAE.replace("1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>",
                             "0 -2 0 0 1 0 0 0 0 0 1 0 0 0 0 1</matrix>")
        source = source.replace("0 0 1</float_array>", "1 1 0</float_array>")
        normal = self.decode(source)[0]["normals"][0]
        self.assertAlmostEqual(normal[0], -1/math.sqrt(5))
        self.assertAlmostEqual(normal[1], 2/math.sqrt(5))
        self.assertEqual(normal[2], 0)
        zero = self.decode()[0]
        zero["triangles"] = ((0, 0, 1),)
        with self.assertRaisesRegex(ValueError, "no visible triangle surface"):
            self.tool["admitted_part"](zero)

    def test_preparation_is_deterministic_and_shared_decoder_admits_output(self):
        part = self.decode()[0]
        first = self.tool["gltf_bytes"](part)
        self.assertEqual(first, self.tool["gltf_bytes"](part))
        from phymeo_rendering.assets import Mesh
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "triangle.gltf"
            path.write_bytes(first)
            geometry = load_mesh(Mesh("owned", path.as_uri()))
            self.assertEqual(geometry.positions, part["positions"])
            self.assertEqual(geometry.normals, part["normals"])
            self.assertEqual(geometry.triangles, ((0, 1, 2),))


class PandaVisualManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))

    def test_real_complete_robot_has_all_links_stable_ids_and_right_finger_pose(self):
        meshes, materials, instances = panda_visual_assets(ASSETS)
        self.assertTrue(all(isinstance(value, tuple) for value in (meshes, materials, instances)))
        expected_links = {f"panda_link{i}" for i in range(8)} | {"panda_hand", "panda_leftfinger", "panda_rightfinger"}
        self.assertEqual({item.node_id for item in instances}, expected_links)
        self.assertEqual(len({item.id for item in instances}), len(instances))
        self.assertGreaterEqual(min(item.id for item in instances), 100)
        self.assertEqual(panda_visual_assets(ASSETS), (meshes, materials, instances))
        for item in instances:
            expected = (0.0, 0.0, 1.0, 0.0) if item.node_id == "panda_rightfinger" else (0.0, 0.0, 0.0, 1.0)
            self.assertEqual(item.local_pose.orientation_xyzw, expected)
            self.assertEqual(item.local_pose.position_m, (0.0, 0.0, 0.0))
        decoded = {mesh.id: load_mesh(mesh) for mesh in meshes}
        self.assertEqual(sum(len(decoded[item.mesh_id].triangles) for item in instances), 134912)
        self.assertEqual({item.material_id for item in instances}, {item.id for item in materials})

    def test_prepared_geometry_hashes_and_declared_counts_match_actual_bytes(self):
        from phymeo_rendering.assets import Mesh
        for record in self.manifest["meshes"]:
            path = ASSETS / record["file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
            result = load_mesh(Mesh(record["id"], path.as_uri()))
            self.assertEqual(len(result.triangles), record["triangles"])
            self.assertEqual(len(result.positions), record["vertices"])

    def test_changed_manifest_missing_link_bad_pose_and_path_escape_rejected(self):
        changes = (
            lambda doc: doc["instances"].pop(),
            lambda doc: doc["instances"][0].update(id=0),
            lambda doc: doc["instances"][-1].update(orientation_xyzw=[0, 0, 0, 1]),
            lambda doc: doc["meshes"][0].update(file="../outside.gltf"),
            lambda doc: doc.update(source_commit="0" * 40),
        )
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as root:
                doc = json.loads(json.dumps(self.manifest))
                change(doc)
                (Path(root) / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
                with self.assertRaises(ValueError):
                    panda_visual_assets(root)

    def test_missing_or_modified_mesh_is_never_replaced_with_proxy(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            shutil.copytree(ASSETS, root / "assets", ignore=shutil.ignore_patterns("source"))
            target = root / "assets" / self.manifest["meshes"][0]["file"]
            original = target.read_bytes()
            target.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "hash|SHA|digest"):
                panda_visual_assets(root / "assets")
            target.unlink()
            with self.assertRaises((OSError, ValueError)):
                panda_visual_assets(root / "assets")

    def test_original_archives_and_zero_area_receipts_cover_every_source_triangle(self):
        import gzip
        for row in self.manifest["sources"]:
            raw = gzip.decompress((ASSETS / row["archive_file"]).read_bytes())
            self.assertEqual(len(raw), row["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])
        for row in self.manifest["meshes"]:
            kept = [i for start, stop in row["source_triangle_runs"] for i in range(start, stop)]
            removed = [item["source_ordinal"] for item in row["omitted_zero_area_triangles"]]
            self.assertEqual(len(kept), row["triangles"])
            self.assertEqual(sorted(kept + removed), list(range(row["source_triangle_count"])))
            self.assertEqual(row["quantization_surface_loss_count"], 0)
        self.assertEqual(self.manifest["totals"]["source_triangles_instanced"], 134924)
        self.assertEqual(self.manifest["totals"]["surface_triangles_instanced"], 134912)
        self.assertEqual(self.manifest["totals"]["source_zero_area_triangles_instanced"], 12)
        self.assertEqual(self.manifest["totals"]["both_fingers_triangles"], 1248)

    def test_urdf_visual_placement_is_bound_to_the_archived_manufacturer_declaration(self):
        import xml.etree.ElementTree as ET
        for row in self.manifest["visual_placement_sources"]:
            self.assertEqual(hashlib.sha256((ASSETS / row["file"]).read_bytes()).hexdigest(), row["sha256"])
        hand = ET.parse(ASSETS / "original" / "franka_hand.xacro")
        links = {link.get("name"): link for link in hand.findall(".//link")}
        left = links["${arm_id}_leftfinger"].find("visual")
        right = links["${arm_id}_rightfinger"].find("visual")
        self.assertIsNone(left.find("origin"))
        self.assertEqual(right.find("origin").attrib, {"xyz": "0 0 0", "rpy": "0 0 ${pi}"})
        for visual in (left, right):
            mesh = visual.find("geometry/mesh")
            self.assertEqual(mesh.attrib, {"filename": "package://${description_pkg}/meshes/visual/finger.dae"})
        utils = ET.parse(ASSETS / "original" / "utils.xacro")
        visual = utils.find(".//visual")
        self.assertIsNone(visual.find("origin"))
        self.assertEqual(visual.find("geometry/mesh").attrib,
                         {"filename": "package://${description_pkg}/meshes/visual/${name}.dae"})

    def test_recipe_reproduces_published_manifest_from_lossless_archives_without_network(self):
        import gzip
        tool = runpy.run_path(str(TOOL))
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source"
            source.mkdir()
            for row in self.manifest["sources"]:
                (source / row["file"]).write_bytes(gzip.decompress((ASSETS / row["archive_file"]).read_bytes()))
            for row in self.manifest["licenses"] + self.manifest["visual_placement_sources"]:
                (source / Path(row["file"]).name).write_bytes((ASSETS / row["file"]).read_bytes())
            output = root / "prepared"
            tool["prepare"](source, output)
            self.assertEqual((output / "manifest.json").read_bytes(), (ASSETS / "manifest.json").read_bytes())
            self.assertEqual(len(panda_visual_assets(output)[2]), 59)

    def test_license_or_notice_modification_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            shutil.copytree(ASSETS, root / "assets", ignore=shutil.ignore_patterns("source"))
            (root / "assets" / "NOTICE").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash|SHA|digest"):
                panda_visual_assets(root / "assets")


if __name__ == "__main__":
    unittest.main()
