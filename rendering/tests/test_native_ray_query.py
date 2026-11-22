"""Explicit hardware shadow mode; recorder tests never claim hardware pixels."""
import ctypes as ct
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from phymeo_rendering.backends import BackendUnavailableError, UnsupportedRequestError
from phymeo_rendering.backends.native import NativeBackend
from test_native_resident import fixture, Recorder


class RayQuerySelection(unittest.TestCase):
    def test_software_default_and_only_explicit_named_hardware_mode(self):
        self.assertEqual(NativeBackend().shadow_mode,'software')
        self.assertEqual(NativeBackend(resident=True,shadow_mode='ray_query').shadow_mode,'ray_query')
        for value in ('hardware','auto','', 'RAY_QUERY'):
            with self.assertRaises(ValueError): NativeBackend(resident=True,shadow_mode=value)
        for value in (None,True,1):
            with self.assertRaises(TypeError): NativeBackend(resident=True,shadow_mode=value)
        with self.assertRaisesRegex(UnsupportedRequestError,'resident'):
            NativeBackend(shadow_mode='ray_query')

    def test_requested_mode_opens_its_device_and_records_the_mode(self):
        scene,mesh=fixture(); hardware=Recorder()
        with patch('phymeo_rendering.backends.native.require_library'), \
             patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
             patch('phymeo_rendering.backends.native.RayQueryResidentDevice',return_value=hardware) as selected, \
             patch('phymeo_rendering.backends.native.ResidentDevice') as software, \
             patch('phymeo_rendering.backends.native.NativeDevice') as legacy:
            owner=NativeBackend(resident=True,shadow_mode='ray_query').open_scene(scene)
            selected.assert_called_once(); software.assert_not_called(); legacy.assert_not_called()
            diagnostics=owner.diagnostics()
            self.assertEqual(diagnostics['shadow_mode'],'ray_query')
            self.assertIn('raster',diagnostics['primary_visibility'])
            self.assertFalse(diagnostics['path_trace'])
            owner.close()
        self.assertTrue(hardware.closed)

    def test_unavailable_hardware_never_falls_back_to_software(self):
        scene,mesh=fixture()
        with patch('phymeo_rendering.backends.native.require_library'), \
             patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
             patch('phymeo_rendering.backends.native.RayQueryResidentDevice',side_effect=BackendUnavailableError('ray query unsupported')), \
             patch('phymeo_rendering.backends.native.ResidentDevice') as software, \
             patch('phymeo_rendering.backends.native.NativeDevice') as legacy:
            with self.assertRaisesRegex(BackendUnavailableError,'ray query'):
                NativeBackend(resident=True,shadow_mode='ray_query').open_scene(scene)
        software.assert_not_called(); legacy.assert_not_called()

    def test_default_resident_does_not_probe_or_require_hardware_extension(self):
        scene,mesh=fixture(); software=Recorder()
        with patch('phymeo_rendering.backends.native.require_library'), \
             patch('phymeo_rendering.mesh_data.load_mesh',return_value=mesh), \
             patch('phymeo_rendering.backends.native.ResidentDevice',return_value=software), \
             patch('phymeo_rendering.backends.native.RayQueryResidentDevice') as hardware:
            owner=NativeBackend(resident=True).open_scene(scene)
            self.assertEqual(owner.diagnostics()['shadow_mode'],'software')
            owner.close()
        hardware.assert_not_called()


class RayQueryABI(unittest.TestCase):
    def test_missing_or_unknown_extension_is_explicit_at_binding(self):
        from phymeo_rendering.backends._native_ray_query import RayQueryResidentDevice
        from phymeo_rendering.backends._native_resident import ResidentDevice
        owner=RayQueryResidentDevice.__new__(RayQueryResidentDevice)
        with patch.object(ResidentDevice,'_bind'):
            owner._lib=SimpleNamespace()
            with self.assertRaisesRegex(BackendUnavailableError,'ray.query'):
                owner._bind()
            owner._lib=SimpleNamespace(pr_resident_ray_query_abi_version=lambda:99,
                pr_resident_create_ray_query=lambda *args:1,pr_resident_ray_query_info=lambda *args:0)
            with self.assertRaisesRegex(BackendUnavailableError,'ray.query'):
                owner._bind()

    def test_hardware_create_uses_dedicated_shader_prefix_and_exact_entry(self):
        from phymeo_rendering.backends._native_ray_query import RayQueryResidentDevice
        owner=RayQueryResidentDevice.__new__(RayQueryResidentDevice); calls=[]
        owner._lib=SimpleNamespace(pr_resident_abi_version=lambda:1)
        owner._ray_query_create=lambda *args:calls.append(args) or 123
        self.assertEqual(owner._shader_prefix,'resident_rayquery')
        self.assertEqual(owner._create(1,1,'vert',20,'frag',24),123)
        self.assertEqual(calls,[(1,1,'vert',20,'frag',24)])

    def test_info_layout_counters_and_last_live_scope_do_not_invent_traces(self):
        from phymeo_rendering.backends._native_ray_query import ResidentRayQueryInfo, RayQueryResidentDevice
        info=ResidentRayQueryInfo(); info.ray_query_abi_version=1; info.enabled=1
        info.blas_count=2; info.tlas_instance_count=3; info.blas_builds=2
        info.tlas_builds=1; info.tlas_updates=3; info.instance_upload_bytes=4*3*64
        info.shadow_captures=4; info.acceleration_buffer_bytes=4096
        owner=RayQueryResidentDevice.__new__(RayQueryResidentDevice)
        owner._remember_ray_query_info(info)
        self.assertEqual(ct.sizeof(info),128)
        self.assertEqual(owner._ray_query_info['instance_upload_bytes'],768)
        self.assertEqual(owner._ray_query_info['tlas_updates'],3)
        self.assertIn('not ray count',owner._ray_query_info['capture_scope'])
        self.assertIn('logical',owner._ray_query_info['buffer_scope'])
        self.assertIn('last live',owner._ray_query_info['snapshot'])
        info.enabled=0
        with self.assertRaises(RuntimeError): owner._remember_ray_query_info(info)
        info.enabled=1; info.ray_query_abi_version=2
        with self.assertRaisesRegex(RuntimeError,'ray.query ABI'): owner._remember_ray_query_info(info)


class RayQueryRecording(unittest.TestCase):
    def runner(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
        import native_multiworld
        return native_multiworld

    def test_recording_cli_passes_explicit_mode_without_changing_the_default(self):
        import contextlib
        import io
        runner=self.runner()
        arguments=['native_multiworld','--source','unused','--output','unused','--library','unused','--shaders','unused']
        for mode,extra in (('software',[]),('ray_query',['--shadow-mode','ray_query'])):
            with patch('sys.argv',arguments+extra), patch.object(runner,'run',return_value={'status':'passed'}), \
                 patch('phymeo_rendering.backends.native.NativeBackend') as factory,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(),0)
                self.assertEqual(factory.call_args.kwargs['shadow_mode'],mode)

    def test_as_instance_upload_is_additional_but_buffer_capacity_is_not_transfer(self):
        runner=self.runner()
        device=dict(resident_captures=1,resident_geometry_upload_bytes=100,
            resident_transform_upload_bytes=256,resident_parameter_upload_bytes=416,
            resident_profile={'tlas_upload_bytes':0},resident_ray_query_abi_version=1,
            resident_ray_query_info={'instance_upload_bytes':128,'acceleration_buffer_bytes':4096,
                                    'scratch_buffer_bytes':4096})
        report=runner.transfer_report({'device':device})
        self.assertEqual(report['host_to_device_bytes'],900)
        self.assertEqual(report['upload_components']['ray_query_instance_upload_bytes'],128)
        device['resident_ray_query_info']=None
        self.assertIsNone(runner.transfer_report({'device':device})['host_to_device_bytes'])

if __name__=='__main__': unittest.main()