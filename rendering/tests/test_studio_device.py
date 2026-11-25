"""Exercise the photographic device policy through its public entry points."""
import ctypes as ct
from contextlib import redirect_stdout
from io import StringIO
import json
import unittest
from unittest.mock import Mock, patch

from phymeo_rendering import StudioRenderer, studio_device
from phymeo_rendering import doctor
from phymeo_rendering import preparation
from phymeo_rendering.backends import BackendUnavailableError
from phymeo_rendering.backends import native
from phymeo_rendering.backends._native_msaa import ResidentMsaaInfo
from phymeo_rendering.scene import VisualScene


def adapter(index=0, kind='discrete', missing=()):
    return {
        'index': index,
        'type': kind,
        'name': f'{kind} adapter {index}',
        'device_uuid': 'a1' * 16,
        'api_version': (1 << 22) | (2 << 12),
        'missing_ray_extensions': list(missing),
    }


class DoctorTests(unittest.TestCase):
    def test_native_failure_returns_nonzero_and_json(self):
        output = StringIO()
        with patch.object(StudioRenderer, 'probe_device', side_effect=RuntimeError('device lost')), redirect_stdout(output):
            status = doctor.main(['--library', 'unused', '--device-index', '0'])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(output.getvalue()), {
            'ok': False, 'diagnostic': {'category': 'native_runtime', 'error': 'device lost'}})


class StudioDeviceTests(unittest.TestCase):
    def setUp(self):
        self.inventory = {
            'loader': 'test-vulkan-loader',
            'loader_api_version': (1 << 22) | (2 << 12),
            'validation_layer': True,
            'adapters': [adapter()],
        }
        self.inspect = self.enterContext(
            patch.object(studio_device, 'inspect_vulkan', return_value=self.inventory))
        self.contexts = []

        def create_context(renderer, scene, index, shadow_mode):
            context = Mock(spec_set=['close', 'info'])
            context.info.return_value = {
                'device_uuid': 'a1' * 16, 'device_name': 'test adapter',
                'resident_max_vertices': 1000, 'resident_max_indices': 3000,
                'resident_max_meshes': 10, 'resident_max_instances': 10,
                'resident_max_instanced_triangles': 1000,
            }
            self.contexts.append(context)
            return context

        self.create_context = self.enterContext(
            patch.object(StudioRenderer, '_candidate_device', autospec=True, side_effect=create_context))
        self.features = {'msaa4': True, 'device_uuid': 'a1' * 16, 'native_device_name': 'discrete adapter 0'}
        self.read_features = self.enterContext(
            patch.object(studio_device, 'resident_capabilities', return_value=self.features))
        self.scene = VisualScene('device-regression', (), (), ())

    def attempted_modes(self):
        return [(call.args[-2], call.args[-1]) for call in self.create_context.call_args_list]

    def test_default_identity_and_cached_report(self):
        renderer = StudioRenderer.for_device()
        first = renderer.probe_device()
        second = renderer.probe_device()
        self.assertEqual(json.loads(json.dumps(first)), second)
        self.assertEqual(first['device_uuid'], self.features['device_uuid'])
        self.assertEqual((renderer.shadow_mode, renderer.msaa_samples), ('ray_query', 4))
        self.assertEqual((first['shadow_mode'], first['msaa_samples']), ('ray_query', 4))
        self.assertEqual(self.attempted_modes(), [(0, 'ray_query')])
        self.inspect.assert_called_once()
        self.read_features.assert_called_once()
        self.contexts[0].close.assert_called_once()

    def test_missing_ray_query_keeps_gpu_shadows_softbox_sampling_and_antialiasing(self):
        self.inventory['adapters'] = [adapter(missing=['VK_KHR_ray_query'])]
        renderer = StudioRenderer.for_device()
        report = renderer.probe_device()
        self.assertEqual(self.attempted_modes(), [(0, 'software')])
        self.assertTrue(renderer.resident)
        self.assertEqual((renderer.shadow_mode, renderer.msaa_samples, renderer.area_light_grid),
                         ('software', 4, 4))
        self.assertFalse(report['ray_query'])
        self.assertTrue(report['msaa4'])
        self.assertTrue(report['degradations'])

    def test_compatible_profile_preserves_msaa_even_on_ray_capable_hardware(self):
        renderer = StudioRenderer.for_device(profile='compatible')
        self.assertEqual((renderer.shadow_mode, renderer.msaa_samples), ('software', 4))
        self.assertEqual(self.attempted_modes(), [(0, 'software')])

    def test_sample_count_falls_to_one_without_losing_ray_query(self):
        self.features['msaa4'] = False
        renderer = StudioRenderer.for_device()
        report = renderer.probe_device()
        self.assertEqual((renderer.shadow_mode, renderer.msaa_samples), ('ray_query', 1))
        self.assertFalse(report['msaa4'])
        self.assertTrue(report['degradations'])
        self.assertEqual(self.attempted_modes(), [(0, 'ray_query')])

    def test_explicit_hardware_or_strict_profile_rejects_missing_ray_query(self):
        self.inventory['adapters'] = [adapter(missing=['VK_KHR_ray_query'])]
        for options in ({'profile': 'strict'}, {'shadow_mode': 'ray_query'}):
            with self.subTest(options=options), self.assertRaises(studio_device.StudioDeviceError) as caught:
                StudioRenderer.for_device(**options)
            self.assertEqual(caught.exception.category, 'ray_query')
        self.create_context.assert_not_called()

    def test_explicit_four_samples_is_a_requirement(self):
        self.features['msaa4'] = False
        for options in ({'profile': 'strict'}, {'msaa_samples': 4}):
            with self.subTest(options=options), self.assertRaises(studio_device.StudioDeviceError) as caught:
                StudioRenderer.for_device(**options)
            self.assertEqual(caught.exception.category, 'msaa')
        self.assertEqual(self.attempted_modes(), [(0, 'ray_query')] * 2)
        for context in self.contexts:
            context.close.assert_called_once()

    def test_base_renderer_is_bound_to_the_selected_hardware(self):
        self.inventory['adapters'] = [adapter(0, 'cpu'), adapter(1, 'integrated')]
        renderer = StudioRenderer.for_device(device_index=1)
        report = renderer.probe_device()
        self.assertEqual((renderer.device_index, report['device_index']), (1, 1))
        self.assertEqual(self.attempted_modes(), [(1, 'ray_query')])

    def test_wrong_uuid_or_unverifiable_name_closes_first_context(self):
        for inventory_uuid, native_name in (('b2' * 16, 'discrete adapter 0'),
                                             (None, 'a different GPU')):
            self.inventory['adapters'] = [dict(adapter(), device_uuid=inventory_uuid)]
            self.features['native_device_name'] = native_name
            with self.subTest(uuid=inventory_uuid), self.assertRaises(studio_device.StudioDeviceError) as caught:
                StudioRenderer.for_device()
            self.assertEqual(caught.exception.category, 'device')
            self.contexts[-1].close.assert_called_once()
        self.assertEqual(self.attempted_modes(), [(0, 'ray_query')] * 2)

    def test_missing_uuid_accepts_a_unique_matching_name_only(self):
        self.inventory['adapters'] = [dict(adapter(), device_uuid=None)]
        self.assertEqual(StudioRenderer.for_device().probe_device()['device_index'], 0)
        self.inventory['adapters'].append(dict(adapter(1), name='discrete adapter 0', device_uuid=None))
        with self.assertRaises(studio_device.StudioDeviceError) as caught:
            StudioRenderer.for_device(device_index=0)
        self.assertEqual(caught.exception.category, 'device')
        self.contexts[-1].close.assert_called_once()

    def test_native_empty_format_result_cannot_lower_quality_after_loader_confirmed_formats(self):
        self.inventory['adapters'][0]['msaa_format_queries_supported'] = True
        self.features.update(msaa4=False, msaa_format_query_empty=True)
        with self.assertRaises(studio_device.StudioDeviceError) as caught:
            StudioRenderer.for_device()
        self.assertEqual(caught.exception.category, 'native_runtime')
        self.assertEqual(self.attempted_modes(), [(0, 'ray_query')])
        self.contexts[0].close.assert_called_once()

    def test_no_icd_and_explicit_cpu_cannot_select_a_replacement(self):
        for adapters, options, category in (([], {}, 'icd'),
                                            ([adapter(0, 'cpu'), adapter(1)], {'device_index': 0}, 'device'),
                                            ([adapter()], {'device_index': 3}, 'device')):
            self.inventory['adapters'] = adapters
            with self.subTest(adapters=adapters, options=options):
                with self.assertRaises(studio_device.StudioDeviceError) as caught:
                    StudioRenderer.for_device(**options)
                self.assertEqual(caught.exception.category, category)
                self.assertEqual(caught.exception.report['category'], category)
                self.create_context.assert_not_called()

    def test_native_allocation_failure_never_selects_another_backend_or_device(self):
        self.inventory['adapters'] = [adapter(), adapter(1)]
        failure = BackendUnavailableError('vkAllocateMemory failed: VK_ERROR_OUT_OF_DEVICE_MEMORY')
        self.create_context.side_effect = failure
        with self.assertRaisesRegex(RuntimeError, 'vkAllocateMemory|OUT_OF_DEVICE_MEMORY'):
            StudioRenderer.for_device()
        self.assertEqual(self.attempted_modes(), [(0, 'ray_query')])
        self.read_features.assert_not_called()

    def test_validation_layer_is_only_required_when_requested(self):
        self.inventory['validation_layer'] = False
        with self.assertRaises(studio_device.StudioDeviceError) as caught:
            StudioRenderer.for_device(validation=True)
        self.assertEqual(caught.exception.category, 'validation')
        self.create_context.assert_not_called()
        renderer = StudioRenderer.for_device(validation=False)
        self.assertFalse(renderer.probe_device()['validation_layer'])
        self.assertEqual(self.attempted_modes(), [(0, 'ray_query')])

    def test_advertised_extensions_with_missing_feature_use_software_shadows(self):
        software_device = Mock(spec_set=['close', 'info'])
        self.create_context.side_effect = [
            BackendUnavailableError('required ray-query/accelerationStructure/bufferDeviceAddress feature unavailable'),
            software_device,
        ]
        renderer = StudioRenderer.for_device()
        self.assertEqual((renderer.shadow_mode, renderer.msaa_samples), ('software', 4))
        self.assertEqual(self.attempted_modes(), [(0, 'ray_query'), (0, 'software')])
        self.read_features.assert_called_once()
        software_device.close.assert_called_once()

    def test_capability_query_failure_closes_the_only_context(self):
        self.read_features.side_effect = BackendUnavailableError('resident capability read failed')
        with self.assertRaisesRegex(RuntimeError, 'resident capability read failed'):
            StudioRenderer.for_device()
        self.assertEqual(self.attempted_modes(), [(0, 'ray_query')])
        self.contexts[0].close.assert_called_once()

    def test_first_prepare_uses_selected_context_for_scene_limits(self):
        renderer = StudioRenderer()
        self.inventory['adapters'] = [adapter(missing=['VK_KHR_ray_query'])]
        with patch.object(native, 'require_library'), patch.object(preparation, 'prepare_for_limits') as prepare:
            self.assertIs(renderer.prepare_scene(self.scene), prepare.return_value)
        self.assertEqual(prepare.call_args.kwargs['geometry_limits']['max_vertices'], 1000)
        self.assertEqual(prepare.call_args.kwargs['device']['uuid'], self.features['device_uuid'])
        self.assertEqual(renderer.probe_device()['msaa_samples'], 4)
        self.assertEqual(self.attempted_modes(), [(0, 'software')])
        self.read_features.assert_called_once()
        self.contexts[0].close.assert_called_once()

    def test_first_open_reuses_selected_context_and_propagates_later_error(self):
        renderer = StudioRenderer()
        self.inspect.assert_not_called()
        self.create_context.assert_not_called()
        self.inventory['adapters'] = [adapter(missing=['VK_KHR_ray_query'])]
        failure = BackendUnavailableError('vkAllocateMemory failed after selecting the device')
        with patch.object(native, 'require_library'), patch.object(native, 'configure_display', side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, 'vkAllocateMemory'):
                renderer.open_scene(self.scene)
        self.assertEqual(self.attempted_modes(), [(0, 'software')])
        self.contexts[0].close.assert_called_once()
        self.assertEqual(renderer.probe_device()['shadow_mode'], 'software')
        self.assertEqual(self.create_context.call_count, 1)

    def test_device_uuid_change_after_probe_rejects_scene_upload(self):
        renderer = StudioRenderer.for_device()
        replacement = Mock(spec_set=['info', 'close'])
        replacement.info.return_value = {'device_uuid': 'b2' * 16}
        self.create_context.side_effect = [replacement]
        with patch.object(native, 'require_library'), patch.object(native, 'pack_scene') as upload:
            with self.assertRaises(studio_device.StudioDeviceError) as caught:
                renderer.open_scene(self.scene)
        self.assertEqual(caught.exception.category, 'device')
        self.assertEqual(self.attempted_modes(), [(0, 'ray_query')] * 2)
        self.inspect.assert_called_once()
        replacement.close.assert_called_once()
        upload.assert_not_called()


class ResidentSampleFormatTests(unittest.TestCase):
    def test_four_samples_requires_color_depth_and_both_dimensions(self):
        device = Mock()
        device.info.return_value = {'device_uuid': '01' * 16, 'device_name': 'GPU',
                                    'max_width': 4096, 'max_height': 4096}
        device._lib.pr_resident_msaa_abi_version.return_value = 1
        cases = (
            (5, 5, 1280, 720, True),
            (1, 5, 1280, 720, False),
            (5, 1, 1280, 720, False),
            (5, 5, 1279, 720, False),
            (5, 5, 1280, 719, False),
        )
        for color, depth, width, height, expected in cases:
            with self.subTest(color=color, depth=depth, width=width, height=height):
                caps = ResidentMsaaInfo(msaa_abi_version=1, shader_compatible=1,
                                        color_sample_counts=color, depth_sample_counts=depth,
                                        max_width=width, max_height=height)

                def read_format_capabilities(handle, output):
                    ct.cast(output, ct.POINTER(ResidentMsaaInfo))[0] = caps
                    return 0

                device._lib.pr_resident_msaa_info.side_effect = read_format_capabilities
                report = studio_device.resident_capabilities(device, (1280, 720), hdr=False)
                self.assertIs(report['msaa4'], expected)
                device._lib.pr_resident_set_msaa.assert_not_called()

    def test_driver_format_check_separates_unsupported_format_from_hard_failure(self):
        for statuses, expected in (((-11, 0, 0), False), ((-11, -2), None),
                                   ((-1,), None), ((0, 0, 0), True)):
            calls = []

            def query(handle, fmt, image_type, tiling, usage, flags, output):
                status = statuses[len(calls)]
                calls.append((fmt, usage))
                if status == 0:
                    ct.cast(output, ct.POINTER(studio_device._ImageFormat)).contents.samples = 5
                return status

            with self.subTest(statuses=statuses):
                result = studio_device._msaa_formats(query, None)
                self.assertEqual(calls, [(97, 0x11), (126, 0x20), (97, 0x15)][:len(statuses)])
                if expected is None:
                    self.assertIn(f'VkResult {statuses[-1]}', result['msaa_query_error'])
                else:
                    self.assertIs(result['msaa_format_queries_supported'], expected)
