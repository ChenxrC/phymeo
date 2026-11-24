"""Hard boundaries, normal-only output and the actual Panda hand housing."""
from dataclasses import replace
from hashlib import sha256
import json
import math
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from preview_stable_lighting.shell_normals import MANIFEST, _smoothed, scene_with_smooth_shells


def tube(sides):
    """An uncapped cylinder with split CAD normals; its middle ring is manifold."""
    xy = [(math.cos(i*2*math.pi/sides), math.sin(i*2*math.pi/sides)) for i in range(sides)]
    p, n, uv = [], [], []
    for i in range(sides):
        angle = (i + .5)*2*math.pi/sides
        face = (math.cos(angle), math.sin(angle), 0.)
        for z in (-1., 0.):
            for di, dz in ((0,0), (1,0), (1,1), (0,0), (1,1), (0,1)):
                p.append((*xy[(i+di) % sides], z+dz))
                n.append(face)
                uv.append(((i+di)/sides, (z+dz+1)/2))
    return np.array(p), np.array(n), np.arange(len(p)).reshape(-1,3), np.array(uv)


class ShellNormals(unittest.TestCase):
    def test_smooth_curve_retains_open_boundaries_and_uv_seam(self):
        p, n, t, uv = tube(24)
        result, _ = _smoothed(p, n, t, uv)
        seam = p[:, 0] == 1  # authored u=0/1 at the same positions
        pinned = (p[:, 2] != 0) | seam
        np.testing.assert_array_equal(result[pinned], n[pinned].astype('<f4'))
        np.testing.assert_allclose(result[~pinned, :2], p[~pinned, :2], atol=1e-6)

    def test_hard_and_nonmanifold_edges_preserve_authored_normals(self):
        p, n, t, _ = tube(8)  # 45-degree creases
        result, stats = _smoothed(p, n, t)
        self.assertEqual(stats['changed_vertices'], 0)
        np.testing.assert_array_equal(result, n.astype('<f4'))
        p, n, t, _ = tube(24)
        # A third incident triangle creates a nonmanifold edge at the middle ring.
        extra = len(p)
        extra_p = np.stack((p[2], p[5], p[2] + (0., 0., .25)))
        p = np.vstack((p, extra_p))
        n = np.vstack((n, np.tile(n[0], (3,1))))
        t = np.vstack((t, [extra, extra+1, extra+2]))
        result, stats = _smoothed(p, n, t)
        pinned = (p == p[2]).all(axis=1) | (p == p[5]).all(axis=1)
        np.testing.assert_array_equal(result[pinned], n[pinned].astype('<f4'))
        self.assertGreater(stats['changed_vertices'], 0)

    def test_real_hand_geometry_identity_cache_and_material_scope(self):
        from phymeo_rendering.assets import Material, Mesh
        from phymeo_rendering.mesh_data import load_mesh
        from phymeo_rendering.scene import Instance, VisualScene

        source = json.loads(MANIFEST.read_bytes())
        chosen = {'panda_visual_hand_01', 'panda_visual_hand_02'}
        meshes = tuple(Mesh(row['id'], (MANIFEST.parent/row['file']).as_uri())
                       for row in source['meshes'] if row['id'] in chosen)
        instances = tuple(Instance(row['id'], row['mesh_id'], row['material_id'], row['node_id'])
                          for row in source['instances'] if row['mesh_id'] in chosen)
        material_ids = {i.material_id for i in instances}
        materials = tuple(Material(row['id'], tuple(row['base_color']), row['metallic'], row['roughness'])
                          for row in source['materials'] if row['id'] in material_ids)
        scene = VisualScene('housing', meshes, materials, instances)
        with TemporaryDirectory() as directory:
            changed, evidence = scene_with_smooth_shells(scene, directory)
            self.assertEqual(evidence['skipped'], {})
            self.assertGreater(evidence['changed_vertices'], 0)
            self.assertEqual(changed.instances, scene.instances)
            self.assertEqual(changed.materials, scene.materials)
            self.assertEqual(changed.meshes[1], meshes[1])  # unpainted insert
            self.assertEqual(changed.meshes[0].id, meshes[0].id)
            before, after = load_mesh(meshes[0]), load_mesh(changed.meshes[0])
            self.assertEqual((before.positions, before.triangles, before.texcoords),
                             (after.positions, after.triangles, after.texcoords))
            self.assertNotEqual(before.normals, after.normals)
            self.assertEqual(evidence['meshes'][0]['source_sha256'], before.source_sha256)
            self.assertEqual(evidence['meshes'][0]['derived_sha256'], after.source_sha256)
            self.assertEqual(scene_with_smooth_shells(scene, directory), (changed, evidence))
            material_variant = replace(scene, materials=tuple(replace(m, roughness=.5) for m in materials))
            untouched, skipped = scene_with_smooth_shells(material_variant, directory)
            self.assertEqual(untouched, material_variant)
            self.assertIn(meshes[0].id, skipped['skipped'])
            self.assertEqual(before.source_sha256,
                             sha256((MANIFEST.parent / 'meshes/hand-01.gltf').read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
