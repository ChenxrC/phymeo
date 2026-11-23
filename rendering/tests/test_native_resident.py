"""Resident native seams; recorder exercises ownership, never claims GPU pixels."""
import ctypes as ct
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.backends import BackendUnavailableError
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene


def fixture():
    scene = VisualScene('resident', (Mesh('shared', 'unused.obj'),),
        (Material('paint', (0.2, 0.3, 0.4), 0.2, 0.5),),
        (Instance(11, 'shared', 'paint', 'a'),
         Instance(12, 'shared', 'paint', 'b', Pose((1., 0., 0.)))))
    mesh = SimpleNamespace(positions=((0.,0.,0.), (1.,0.,0.), (0.,1.,0.)),
                          normals=((0.,0.,1.),)*3, triangles=((0,1,2),))
    return scene, mesh


def frame(revision=0):
    key = FrameKey('resident', 'world', 0, revision, revision, 'authored')
    return SceneFrame(key, float(revision),
        (NodePose('a', Pose((0.,0.,2.+revision))),
         NodePose('b', Pose((2.,0.,2.), (0.,0.,math.sqrt(.5),math.sqrt(.5))))))


class Recorder:
    def __init__(self, *args):
        self.uploads = []; self.transforms = []; self.closed = False
        self.library_path = Path('recorder.dll'); self.library_sha256 = 'test'
        self.shader_sha256 = {}
    def info(self):
        return dict(max_width=4096, max_height=4096, max_vertices=2000000,
                    resident_max_instanced_triangles=1048576)
    def upload(self, *args):
        self.uploads.append(args)
    def render(self, transforms, parameters, width, height):
        import struct
        self.transforms.append(tuple(transforms))
        count = width*height
        return bytes(count*4) + struct.pack('<f', math.inf)*count + bytes(count*5)
    def close(self): self.closed = True


class ResidentNative(unittest.TestCase):
    def test_submesh_transform_reuse_tracks_the_frame_and_full_local_pose(self):
        from dataclasses import replace
        from phymeo_rendering.backends.native import _instance_matrices

        scene, _ = fixture()
        first, second = scene.instances
        scene = replace(scene, instances=(
            first, replace(first, id=13), second,
            replace(first, id=14, local_pose=Pose((0., 2., 0.))),
        ))
        for revision in (0, 3, 0):
            value = frame(revision)
            actual = bytes(_instance_matrices(scene, value))
            individual = b"".join(bytes(_instance_matrices(replace(scene, instances=(i,)), value))
                                  for i in scene.instances)
            self.assertEqual(actual, individual)
            self.assertNotEqual(actual[:64], actual[-64:])

    def test_opt_in_preserves_default_and_rejects_non_bool(self):
        self.assertFalse(NativeBackend().resident)
        self.assertTrue(NativeBackend(resident=True).resident)
        with self.assertRaises(TypeError): NativeBackend(resident=1)

    def test_unique_indexed_geometry_and_material_instances(self):
        from phymeo_rendering.backends._native_resident import pack_scene
        scene, mesh = fixture()
        vertices, indices, meshes, instances = pack_scene(scene, {'shared':mesh})
        self.assertEqual((len(vertices),len(indices),len(meshes),len(instances)), (3,3,1,2))
        self.assertEqual(tuple(indices), (0,1,2))
        self.assertEqual([i.mesh_index for i in instances], [0,0])
        self.assertEqual([i.instance_id for i in instances], [11,12])
        self.assertAlmostEqual(instances[1].material[0], .2)
        self.assertEqual(tuple(vertices[1].position), (1.,0.,0.,1.))

    def test_local_pose_and_node_rotation_are_composed_column_major(self):
        from phymeo_rendering.backends.native import _instance_matrices
        scene, _ = fixture()
        values = _instance_matrices(scene, frame())
        self.assertEqual(len(values),32)
        self.assertEqual(tuple(values[12:16]), (0.,0.,2.,1.))
        for actual, expected in zip(values[28:32], (2.,1.,2.,1.)):
            self.assertAlmostEqual(actual,expected,places=6)
        for actual, expected in zip(values[16:20], (0.,1.,0.,0.)):
            self.assertAlmostEqual(actual,expected,places=6)

    def test_public_session_uploads_once_then_only_transforms(self):
        scene, mesh = fixture(); device = Recorder()
        with patch('phymeo_rendering.backends.native.require_library'), \
             patch('phymeo_rendering.mesh_data.load_mesh', return_value=mesh) as decode, \
             patch('phymeo_rendering.backends.native.ResidentDevice', return_value=device), \
             patch('phymeo_rendering.backends.native._vertices', side_effect=AssertionError('legacy expansion')):
            session = NativeBackend(resident=True).open_scene(scene)
            for value in (frame(), frame(1)):
                camera = Camera('camera',value.key,Pose(),2,2,2.,2.,1.,1.,.1,10.)
                result = session.capture(value,CaptureRequest(camera,RenderPath.RASTER,(Channel.RGB,)))
                self.assertEqual(result.frame_key,value.key)
            self.assertEqual(decode.call_count,1)
            self.assertEqual(len(device.uploads),1)
            self.assertEqual(len(device.transforms),2)
            self.assertNotEqual(device.transforms[0],device.transforms[1])
            self.assertEqual(session.diagnostics()['geometry_residency'], 'static indexed meshes; per-frame instance matrices')
            session.close(); self.assertTrue(device.closed)

    def test_resident_unavailable_does_not_open_legacy_device(self):
        scene, mesh = fixture()
        with patch('phymeo_rendering.backends.native.require_library'), \
             patch('phymeo_rendering.mesh_data.load_mesh', return_value=mesh), \
             patch('phymeo_rendering.backends.native.ResidentDevice', side_effect=BackendUnavailableError('resident ABI')), \
             patch('phymeo_rendering.backends.native.NativeDevice') as legacy:
            with self.assertRaisesRegex(BackendUnavailableError,'resident ABI'):
                NativeBackend(resident=True).open_scene(scene)
            legacy.assert_not_called()

    def test_upload_failure_closes_real_owner(self):
        scene, mesh = fixture(); device = Recorder()
        device.upload = lambda *args: (_ for _ in ()).throw(RuntimeError('upload failed'))
        with patch('phymeo_rendering.backends.native.require_library'), \
             patch('phymeo_rendering.mesh_data.load_mesh', return_value=mesh), \
             patch('phymeo_rendering.backends.native.ResidentDevice', return_value=device):
            with self.assertRaisesRegex(RuntimeError,'upload failed'):
                NativeBackend(resident=True).open_scene(scene)
        self.assertTrue(device.closed)

    def test_fp32_geometry_and_transform_overflow_rejected(self):
        from phymeo_rendering.backends._native_resident import pack_scene
        from phymeo_rendering.backends.native import _instance_matrices
        scene, mesh = fixture()
        mesh.positions = ((1e100,0.,0.),)+mesh.positions[1:]
        with self.assertRaises(ValueError): pack_scene(scene, {'shared':mesh})
        f = frame()
        bad = SceneFrame(f.key,f.time_s,(NodePose('a',Pose((1e100,0.,0.))),f.nodes[1]))
        with self.assertRaises(ValueError): _instance_matrices(scene,bad)


class ResidentCapacity(unittest.TestCase):
    def test_more_than_legacy_budget_uses_resident_without_legacy_device(self):
        scene, mesh = fixture(); mesh.triangles = ((0,1,2),)*3000
        device = Recorder()
        with patch('phymeo_rendering.backends.native.require_library'), \
             patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
             patch('phymeo_rendering.backends.native.ResidentDevice',return_value=device), \
             patch('phymeo_rendering.backends.native.NativeDevice') as legacy:
            session = NativeBackend(resident=True).open_scene(scene)
            self.assertEqual(session.diagnostics()['triangles'],6000)
            self.assertEqual(len(device.uploads[0][0]),3)
            legacy.assert_not_called()
            session.close()

    def test_device_limit_rejects_before_buffer_allocation(self):
        from phymeo_rendering.backends._native_resident import pack_scene
        from phymeo_rendering.backends import UnsupportedRequestError
        scene, mesh = fixture()
        with self.assertRaisesRegex(UnsupportedRequestError,'max_vertices'):
            pack_scene(scene,{'shared':mesh},{'resident_max_vertices':2})

    def test_mesh_local_indices_must_not_cross_mesh_boundaries(self):
        from phymeo_rendering.backends._native_resident import pack_scene
        scene, mesh = fixture(); mesh.triangles = ((0,1,3),)
        with self.assertRaisesRegex(ValueError,'index outside'):
            pack_scene(scene,{'shared':mesh})

    def test_empty_scene_has_four_empty_upload_tables(self):
        from phymeo_rendering.backends._native_resident import pack_scene
        packed = pack_scene(VisualScene('empty',(),(),()),{})
        self.assertEqual(tuple(map(len,packed)),(0,0,0,0))


class ResidentTelemetry(unittest.TestCase):
    def test_layout_and_adapter_luid_are_exact(self):
        from phymeo_rendering.backends._native_resident import ResidentTelemetry, ResidentDevice
        value=ResidentTelemetry(); value.telemetry_abi_version=1
        value.device_luid_valid=1; value.device_luid[:]=bytes.fromhex('7856341201000000')
        value.device_local_bytes_current=4096; value.fence_wait_wall_ns=123
        owner=ResidentDevice.__new__(ResidentDevice)
        owner._remember_telemetry(value,final=False)
        self.assertEqual(ct.sizeof(value),128)
        self.assertEqual(owner._telemetry['device_luid_hex'],'7856341201000000')
        self.assertEqual(owner._telemetry['device_local_bytes_current'],4096)
        self.assertFalse(owner._telemetry['final_after_destroy'])

    def test_unknown_telemetry_version_cannot_be_mislabelled(self):
        from phymeo_rendering.backends._native_resident import ResidentTelemetry, ResidentDevice
        value=ResidentTelemetry(); value.telemetry_abi_version=2
        owner=ResidentDevice.__new__(ResidentDevice)
        with self.assertRaisesRegex(RuntimeError,'telemetry ABI'):
            owner._remember_telemetry(value,final=False)

    def test_optional_destroy_returns_real_final_allocation_counters(self):
        from phymeo_rendering.backends._native_resident import ResidentTelemetry, ResidentDevice
        from phymeo_rendering.backends._native_device import _Info
        calls=[]
        def destroy(handle,info,telemetry):
            calls.append(handle)
            ct.cast(info,ct.POINTER(_Info)).contents.captures=2
            value=ct.cast(telemetry,ct.POINTER(ResidentTelemetry)).contents
            value.telemetry_abi_version=1; value.device_local_bytes_peak=8192
            value.allocation_count=5; value.free_count=5
            return 0
        owner=ResidentDevice.__new__(ResidentDevice)
        owner._handle=123; owner._resident_destroy=destroy; owner._last_info={}
        owner.info=lambda: {}
        owner.close(); owner.close()
        self.assertEqual(calls,[123])
        self.assertEqual(owner._telemetry['device_local_bytes_current'],0)
        self.assertEqual(owner._telemetry['device_local_bytes_peak'],8192)
        self.assertTrue(owner._telemetry['final_after_destroy'])


class ResidentProfileContract(unittest.TestCase):
    def test_optional_profile_layout_and_times_keep_their_real_scope(self):
        from phymeo_rendering.backends._native_resident import ResidentProfile, ResidentDevice
        value=ResidentProfile(); value.profile_abi_version=1; value.timestamp_supported=1
        value.timestamp_valid_bits=64; value.timestamp_period_ns=1.
        value.captures=3; value.tlas_upload_bytes=432; value.gpu_render_ns=1000
        value.timestamp_samples=3
        owner=ResidentDevice.__new__(ResidentDevice); owner._remember_profile(value)
        self.assertEqual(ct.sizeof(value),128)
        self.assertEqual(owner._profile['tlas_upload_bytes'],432)
        self.assertEqual(owner._profile['gpu_render_ns'],1000)
        self.assertIn('excludes initial static',owner._profile['scope'])
    def test_unknown_profile_version_rejected(self):
        from phymeo_rendering.backends._native_resident import ResidentProfile, ResidentDevice
        owner=ResidentDevice.__new__(ResidentDevice); value=ResidentProfile()
        value.profile_abi_version=99
        with self.assertRaisesRegex(RuntimeError,'profile ABI'): owner._remember_profile(value)


if __name__ == '__main__': unittest.main()
