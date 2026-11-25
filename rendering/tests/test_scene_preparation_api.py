"""Camera priorities, immutable source bindings and the public preparation entry."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from phymeo_rendering import PreparedScene, StudioRenderer
from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.backends import UnsupportedRequestError
from phymeo_rendering.backends._native_resident import _LIMITS
from phymeo_rendering.scene import FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene


class ScenePreparationApiTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / 'copy.gltf'
        self.path.write_bytes((Path(__file__).parent / 'fixtures/g1/block.gltf').read_bytes())
        meshes = tuple(Mesh(name, self.path.as_uri()) for name in ('near', 'far'))
        paint = Material('paint', (.3, .4, .5), 0., .5)
        instances = (Instance(1, 'near', 'paint', 'arm'), Instance(2, 'far', 'paint', 'wall'))
        self.scene = VisualScene('input', meshes, (paint,), instances)
        key = FrameKey('input', 'robot', 3, 7, 8, 'accepted_endpoint')
        self.frame = SceneFrame(key, .25, (NodePose('arm', Pose((0., 0., -1.))),
                                          NodePose('wall', Pose((0., 0., -10.)))))
        self.renderer = StudioRenderer()
        self.cam = self.renderer.camera('main', key, Pose())
        self.device = Mock()
        info = {'resident_' + k: v for k, v in _LIMITS.items()}
        self.device.info.return_value = dict(info, device_name='test', device_uuid='test')
        for mocked in (patch('phymeo_rendering.backends.native.require_library'),
                       patch.object(self.renderer, '_scene_device', return_value=self.device)):
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_public_exact_mode_preserves_exposure_and_returns_independent_report(self):
        result = self.renderer.prepare_scene(self.scene, cameras=[(self.frame, self.cam)], preserve_detail=['arm'])
        self.assertIsInstance(result, PreparedScene)
        self.assertIs(result.scene, self.scene)
        bound = result.bind_view(self.frame, self.cam)
        self.assertIs(bound[0], self.frame)
        self.assertIs(bound[1], self.cam)
        info = result.report
        self.assertEqual(info['usage'], 'sensor')
        self.assertEqual(info['protected_meshes'], ['near'])
        self.assertEqual(info['protected_materials'], ['paint'])
        spans = info['maximum_projected_span_px']
        self.assertGreater(spans['near'], spans['far'])
        del info['geometry']
        self.assertIn('geometry', result.report)
        self.device.close.assert_called_once()

    def test_independent_geometry_and_material_protection(self):
        plan = self.renderer.prepare_scene(self.scene, usage='film', preserve_meshes=['near'])
        self.assertEqual(plan.report['protected_meshes'], ['near'])
        self.assertEqual(plan.report['protected_materials'], [])
        plan = self.renderer.prepare_scene(self.scene, usage='film', preserve_materials=['paint'])
        self.assertEqual(plan.report['protected_meshes'], [])
        self.assertEqual(plan.report['protected_materials'], ['paint'])
        with self.assertRaisesRegex(ValueError, 'unknown meshes'):
            self.renderer.prepare_scene(self.scene, usage='film', preserve_meshes=['arm'])

    def test_additional_closeup_increases_shared_resource_priority(self):
        shared = replace(self.scene, instances=self.scene.instances + (Instance(3, 'far', 'paint', 'wall-copy'),))
        frame = replace(self.frame, nodes=self.frame.nodes + (NodePose('wall-copy', Pose((20., 0., -1.))),))
        cam2 = replace(self.cam, id='wrist', world_pose=Pose((20., 0., -.5)))
        a = self.renderer.prepare_scene(shared, usage='film', cameras=[(frame, self.cam)])
        b = self.renderer.prepare_scene(shared, usage='film', cameras=[(frame, self.cam), (frame, cam2)])
        self.assertGreater(b.report['maximum_projected_span_px']['far'], a.report['maximum_projected_span_px']['far'])
        self.assertEqual(b.report['geometry']['before']['max_meshes'], 2)
        self.assertEqual(b.report['geometry']['before']['max_instances'], 3)

    def test_strict_protection_and_source_validation(self):
        self.device.info.return_value['resident_max_indices'] = 1
        with self.assertRaises(UnsupportedRequestError):
            self.renderer.prepare_scene(self.scene)
        with self.assertRaises(UnsupportedRequestError):
            self.renderer.prepare_scene(self.scene, usage='film', preserve_detail=('paint',))
        with self.assertRaisesRegex(ValueError, 'unknown detail'):
            self.renderer.prepare_scene(self.scene, preserve_detail=('absent',))
        other = replace(self.cam, frame_key=replace(self.cam.frame_key, revision=9))
        with self.assertRaisesRegex(ValueError, 'revision'):
            self.renderer.prepare_scene(self.scene, cameras=[(self.frame, other)])

    def test_rebinding_changes_only_scene_identity_and_detects_modified_source(self):
        result = self.renderer.prepare_scene(self.scene)
        alternate = replace(result, scene=replace(result.scene, key='derived'))
        frame, camera = alternate.bind_view(self.frame, self.cam)
        self.assertEqual(frame.key, replace(self.frame.key, scene_key='derived'))
        self.assertIs(frame.nodes, self.frame.nodes)
        self.assertEqual(camera, replace(self.cam, frame_key=frame.key))
        with self.assertRaisesRegex(ValueError, 'source scene'):
            alternate.bind_view(frame, camera)
        self.path.write_bytes(self.path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'changed before upload'):
            self.renderer.open_scene(result)

    def test_device_closed_on_query_failure(self):
        self.device.info.side_effect = RuntimeError('device query failed')
        with self.assertRaisesRegex(RuntimeError, 'device query'):
            self.renderer.prepare_scene(self.scene)
        self.device.close.assert_called_once()
