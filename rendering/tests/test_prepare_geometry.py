"""Native scene geometry preparation, tested without opening a renderer."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from phymeo_rendering import _prepare_geometry as g
from phymeo_rendering.assets import Material, Mesh, TexturedMaterial, TextureMap
from phymeo_rendering.backends import UnsupportedRequestError
from phymeo_rendering.backends._native_resident import _limits, pack_scene
from phymeo_rendering.mesh_data import load_mesh
from phymeo_rendering.scene import Instance, VisualScene


class GeometryPreparationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cache = self.root / 'cache'
        self.paint = Material('paint', (.2, .3, .4), 0., .4)

    def mesh(self, name, *, unused=0):
        # Two independent UV islands meet along the middle of this sheet.
        # Preserve the duplicated edge with its different UV tuples.
        pos, uv, triangles = [], [], []
        for island in range(2):
            start = len(pos)
            for y in range(9):
                for x in range(9):
                    pos.append((island + x / 8, y / 8, 0.))
                    uv.append((x / 8, y / 8))
            for y in range(8):
                for x in range(8):
                    a = start + y * 9 + x
                    triangles.extend(((a, a+1, a+10), (a, a+10, a+9)))
        pos.extend([(2., 1., 0.)] * unused)
        uv.extend([(1., 1.)] * unused)
        p = np.asarray(pos, dtype='<f4')
        data = g._mesh_bytes(p, np.tile([0., 0., 1.], (len(p), 1)), np.asarray(uv), np.asarray(triangles))
        path = self.root / (name + '.gltf')
        path.write_bytes(data)
        return Mesh(name, path.as_uri())

    def scene(self, meshes):
        instances = tuple(Instance(i+1, m.id, 'paint', f'node-{i}') for i, m in enumerate(meshes))
        return VisualScene('test', tuple(meshes), (self.paint,), instances)

    def fit(self, scene, *, protected=(), weights=None, adapt=True, **limits):
        return g.fit_geometry(scene, limits=dict(_limits(), **limits), protected_meshes=set(protected),
                              importance={} if weights is None else weights, allow_adapt=adapt, cache=self.cache)

    def test_passthrough_under_budget_and_native_costs(self):
        mesh = self.mesh('tile')
        scene = self.scene([mesh])
        scene = replace(scene, instances=scene.instances + (Instance(10, mesh.id, 'paint', 'second'),))
        received = []
        def priority(asset, decoded):
            received.append((asset, decoded.source_sha256))
            return 1.
        with patch.object(g, '_adapt', side_effect=AssertionError('exact case called reducer')):
            result, report = self.fit(scene, weights=priority)
        self.assertIs(result, scene)
        self.assertFalse(self.cache.exists())
        self.assertEqual(received, [(mesh, sha256((self.root/'tile.gltf').read_bytes()).hexdigest())])
        counts = report['after']
        v, indices, meshes, instances = pack_scene(scene, {mesh.id: load_mesh(mesh)})
        self.assertEqual((counts['max_vertices'], counts['max_indices'], counts['max_meshes'], counts['max_instances']),
                         (len(v), len(indices), len(meshes), len(instances)))
        self.assertEqual(counts['max_instanced_triangles'], 2*len(indices)//3)

    def test_explicit_limits_and_immutable_slots(self):
        scene = self.scene([self.mesh('m')])
        for cap in ('max_meshes', 'max_instances'):
            with self.subTest(cap=cap), self.assertRaisesRegex(UnsupportedRequestError, cap):
                self.fit(scene, **{cap: 0})
        with self.assertRaisesRegex(UnsupportedRequestError, 'exact.*max_indices'):
            self.fit(scene, adapt=False, max_indices=765)
        with self.assertRaisesRegex(UnsupportedRequestError, 'protected.*max_indices'):
            self.fit(scene, protected=('m',), max_indices=765)
        with self.assertRaisesRegex(ValueError, 'outside the scene'):
            self.fit(scene, weights={'other': 2})
        with self.assertRaisesRegex(ValueError, 'importance'):
            self.fit(scene, weights={'m': 0})

    def test_vertex_only_overflow_compacts_without_vtk(self):
        mesh = self.mesh('m', unused=20)
        scene = self.scene([mesh])
        with patch.object(g, '_adapt', side_effect=AssertionError('unused vertices need no decimation')):
            result, report = self.fit(scene, max_vertices=162)
        self.assertEqual(report['meshes']['m']['action'], 'compact')
        self.assertEqual(len(load_mesh(result.meshes[0]).triangles), 256)
        self.assertEqual(result.instances, scene.instances)
        self.assertEqual(result.meshes[0].id, mesh.id)

    def test_mapped_seams_priority_instances_cache_and_distance(self):
        high, low = self.mesh('important'), self.mesh('other')
        mat = TexturedMaterial('paint', (.2,.3,.4), 0., .4, normal_map=TextureMap('file:///dummy.png', '0'*64))
        scene = replace(self.scene([high,low]), materials=(mat,),
                        instances=(Instance(1,'important','paint','a'), Instance(2,'other','paint','b'),
                                   Instance(3,'other','paint','c')))
        def run():
            return self.fit(scene, weights={'important': 3}, max_instanced_triangles=480)
        out, report = run()
        self.assertEqual(out.instances, scene.instances)
        self.assertEqual(out.materials, scene.materials)
        self.assertEqual(out.meshes[0], high)
        rows = report['meshes']
        self.assertLess(rows['other']['after']['triangles'], rows['important']['after']['triangles'])
        self.assertLessEqual(report['after']['max_instanced_triangles'], 480)
        mesh = load_mesh(out.meshes[1])
        self.assertEqual(len(mesh.positions), rows['other']['after']['vertices'])
        self.assertEqual(mesh.source_sha256, rows['other']['output_sha256'])
        self.assertLess(rows['other']['distance']['source_to_lod']['max_m'], 1e-6)
        border = set(zip(mesh.positions, mesh.texcoords))
        for y in range(9):
            point = (1., y/8, 0.)
            self.assertIn((point, (0., y/8)), border)
            self.assertIn((point, (1., y/8)), border)
        tri = np.asarray(mesh.triangles)
        self.assertFalse(g._invalid_faces(np.asarray(mesh.positions), np.asarray(mesh.texcoords), tri, True).any())
        with patch.object(g, '_simplify', side_effect=AssertionError('cache was not reused')):
            again, other_report = run()
        self.assertEqual(again, out)
        self.assertEqual(other_report, report)
