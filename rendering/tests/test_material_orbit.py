"""Authored camera-only material comparison; these tests make no GPU quality claim."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
sys.path.insert(0, str(EXAMPLES))
SPEC = importlib.util.spec_from_file_location('material_orbit', EXAMPLES / 'material_orbit.py')
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


class MaterialOrbitTests(unittest.TestCase):
    def test_full_clip_has_eight_seconds_of_camera_only_authored_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene, nodes = subject.build_scene(Path(temporary))
            sequence = subject.exposures(scene, nodes, fps=24)
            self.assertEqual(len(sequence), 192)
            self.assertEqual(sequence[0][0].time_s, 0.)
            self.assertEqual(sequence[-1][0].time_s, 191/24)
            self.assertTrue(all(frame.nodes == nodes for frame, camera in sequence))
            self.assertEqual(len({camera.world_pose for _, camera in sequence}), 192)
            self.assertTrue(all(frame.key.phase == 'authored' and frame.key.revision == 0
                                and camera.frame_key == frame.key for frame, camera in sequence))
            self.assertEqual(subject.scout_indices(192), (0, 96, 191))

    def test_same_shape_two_metals_and_rubber_separate_scanned_uv_board(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene, nodes = subject.build_scene(Path(temporary))
            instances = {item.id: item for item in scene.instances}
            self.assertEqual(len({instances[i].mesh_id for i in (1, 2, 3)}), 1)
            materials = {item.id: item for item in scene.materials}
            steel, rough, rubber = [materials[instances[i].material_id] for i in (1, 2, 3)]
            self.assertEqual(steel.base_color, rough.base_color)
            self.assertEqual((steel.metallic, rough.metallic, rubber.metallic), (1., 1., 0.))
            self.assertEqual((steel.roughness, rough.roughness, rubber.roughness), (.035, .45, .85))
            wood = materials[instances[4].material_id]
            self.assertEqual(wood.base_color_map.sha256, subject.SOURCES['wood_table_001_diff_1k.jpg'])
            self.assertEqual(wood.roughness_map.image.sha256, subject.SOURCES['wood_table_001_rough_1k.jpg'])
            self.assertEqual(wood.roughness_map.channel, 'r')
            from phymeo_rendering.mesh_data import load_mesh
            board = load_mesh(next(mesh for mesh in scene.meshes if mesh.id == instances[4].mesh_id))
            self.assertEqual(len(board.texcoords), len(board.positions))

    def test_predeclared_angles_project_all_four_centres_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene, nodes = subject.build_scene(Path(temporary))
            sequence = subject.exposures(scene, nodes)
            for index in subject.scout_indices(len(sequence)):
                _, camera = sequence[index]
                projected = [subject.project(camera, subject.CENTRES[i]) for i in (1, 2, 3, 4)]
                self.assertTrue(all(70 < x < camera.width-70 and 70 < y < camera.height-70
                                    and camera.near_m < z < camera.far_m for x,y,z in projected))
                self.assertTrue(all(b[0]-a[0] > 75 for a,b in zip(projected, projected[1:])))

    def test_mesh_has_nonzero_area_outward_triangles_and_unit_normals(self):
        positions, normals, triangles = subject.sphere_mesh()
        p, n, t = np.asarray(positions), np.asarray(normals), np.asarray(triangles)
        area = np.cross(p[t[:,1]]-p[t[:,0]], p[t[:,2]]-p[t[:,0]])
        self.assertTrue(np.all(np.linalg.norm(area,axis=1) > 1e-10))
        self.assertTrue(np.all(np.sum(area*p[t].mean(axis=1),axis=1) > 0))
        np.testing.assert_allclose(np.linalg.norm(n,axis=1), 1., atol=1e-14)

    def test_asset_digest_refuses_modified_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'wood_table_001_diff_1k.jpg').write_bytes(b'not the declared source')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                subject.verify_sources(root)

    def test_bounded_frame_rate_and_resolution_reject_invalid_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene, nodes = subject.build_scene(Path(temporary))
            for fps in (0, True, 24.5, 120):
                with self.subTest(fps=fps), self.assertRaises((TypeError, ValueError)):
                    subject.exposures(scene,nodes,fps=fps)
            with self.assertRaises(ValueError):
                subject.exposures(scene,nodes,width=1280,height=0)


if __name__ == '__main__':
    unittest.main()
