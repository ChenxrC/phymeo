"""Public session ownership gates. The real raster comparison lives in examples."""
import ctypes as ct
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from phymeo_rendering.backends import BackendUnavailableError, SessionClosedError, UnsupportedRequestError
from phymeo_rendering.backends.native import DevicePoseExposure, NativeBackend, NativeRenderSession
from phymeo_rendering.scene import FrameKey, Pose, VisualScene


class DevicePoseSessionTests(unittest.TestCase):
    def setUp(self):
        self.scene = VisualScene("fixture", (), (), ())
        self.delegate = Mock()
        self.session = NativeRenderSession(self.scene, self.delegate, shared_cuda_device=0)

    def test_exact_exposure_identity_required(self):
        key = FrameKey("fixture", "41", 0, 1, 0, "accepted_endpoint")
        token = DevicePoseExposure(key, 0.0)
        other = DevicePoseExposure(key, 0.0)
        self.session._active_device_poses = SimpleNamespace(exposure=token)
        with self.assertRaisesRegex(ValueError, "foreign or expired"):
            self.session.capture_device_poses(other, None)
        with self.assertRaisesRegex(RuntimeError, "before closing"):
            self.session.close()
        self.delegate.capture.assert_not_called()
        self.delegate.close.assert_not_called()

    def test_mount_metadata_reads_once_and_uses_each_quaternion_branch(self):
        from phymeo_rendering.backends.native import _rotation
        from phymeo_rendering.device_pose_packing import _metadata_pose

        token = DevicePoseExposure(FrameKey("fixture", "41", 0, 1, 0, "accepted_endpoint"), 0.)
        binding = Mock()
        binding.camera_slot.return_value = 0
        active = SimpleNamespace(exposure=token, workspace=Mock(), frame=object(), binding=binding, packed=object())
        self.session._active_device_poses = active
        lens = dict(width=4, height=4, fx_px=4, fy_px=4, cx_px=2, cy_px=2, near_m=.01, far_m=2.)
        for quat in ((1.,0.,0.,0.), (0.,1.,0.,0.), (0.,0.,1.,0.), (0.,0.,0.,1.), (.5,-.5,.5,.5)):
            pose = Pose((1.,2.,3.), quat)
            rows = [(*row, p) for row, p in zip(_rotation(pose), pose.position_m, strict=True)]
            metadata = _metadata_pose(rows)
            self.assertEqual(metadata.position_m, pose.position_m)
            self.assertEqual(_rotation(metadata), _rotation(pose))
            binding.camera_metadata.reset_mock()
            binding.camera_metadata.return_value = metadata
            active.camera_poses = {}
            for _ in range(2):
                camera = self.session.mounted_camera(token, "mount", **lens)
                self.assertEqual(camera.world_pose, metadata)
            binding.camera_metadata.assert_called_once()

    def test_retry_close_after_retained_fence_timeout(self):
        self.delegate.close.side_effect = [TimeoutError("pending fence"), None]
        with self.assertRaises(TimeoutError):
            self.session.close()
        with self.assertRaises(SessionClosedError):
            self.session.capture_device_poses(None, None)
        self.session.close()
        self.session.close()
        self.assertEqual(self.delegate.close.call_count, 2)

    def test_unsupported_modes_and_host_capture(self):
        for opts in (dict(resident=False), dict(resident=True, area_light_grid=8),
                     dict(resident=True, shadow_mode="ray_query", indirect_mode="diffuse_bounce")):
            with self.subTest(opts=opts), self.assertRaises(UnsupportedRequestError):
                NativeBackend(shared_cuda_device=0, **opts)
        with self.assertRaisesRegex(UnsupportedRequestError, "capture_device_poses"):
            self.session.capture(None, None)
        with self.assertRaises(UnsupportedRequestError):
            with NativeRenderSession(self.scene, self.delegate).publish_device_poses(
                        None, None, None, producer_stream=None):
                self.fail("host-only session unexpectedly acquired shared poses")

    def test_shared_modes_and_optional_mounted_camera_capacity(self):
        for mode in ("software", "ray_query"):
            for samples in (1,4):
                with self.subTest(mode=mode, samples=samples):
                    owner = NativeBackend(resident=True, shared_cuda_device=0, shadow_mode=mode,
                                          msaa_samples=samples, shared_camera_capacity=64)
                    self.assertEqual((owner.shadow_mode,owner.msaa_samples,owner.shared_camera_capacity),
                                     (mode,samples,64))
        self.assertEqual(NativeBackend().shared_camera_capacity, 0)
        for invalid in (-1, 65, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "shared_camera_capacity"):
                NativeBackend(resident=True, shared_cuda_device=0, shared_camera_capacity=invalid)
        with self.assertRaisesRegex(ValueError, "requires a shared CUDA"):
            NativeBackend(resident=True, shared_camera_capacity=1)


class SharedDeviceExtensionTests(unittest.TestCase):
    def test_software_msaa_upgrade_requires_new_abi_without_changing_1x(self):
        from phymeo_rendering.backends._native_msaa import configure_msaa

        device = SimpleNamespace(_shared_cuda_device=0, _lib=SimpleNamespace())
        self.assertEqual(configure_msaa(device, 1)["actual_color_samples"], 1)
        with self.assertRaisesRegex(BackendUnavailableError, "missing pr_resident_shared_modes"):
            configure_msaa(device, 4)
        device._lib.pr_resident_shared_modes_abi_version = Mock(return_value=2)
        with self.assertRaisesRegex(BackendUnavailableError, "ABI 2"):
            configure_msaa(device, 4)

    def test_hardware_shared_create_requires_new_symbol_and_actual_compute_module(self):
        from phymeo_rendering.backends._native_ray_query import RayQueryResidentDevice
        from phymeo_rendering.backends._native_resident import ResidentDevice

        device = RayQueryResidentDevice.__new__(RayQueryResidentDevice)
        lib = SimpleNamespace(pr_create=Mock(argtypes=[]), pr_resident_create=Mock(argtypes=[]),
                              pr_resident_create_ray_query=Mock(), pr_resident_ray_query_info=Mock(),
                              pr_resident_ray_query_abi_version=Mock(return_value=1),
                              pr_resident_abi_version=Mock(return_value=1))
        device._lib = lib
        device._shared_identity = (None,None,bytes(range(16)))
        with patch.object(ResidentDevice, "_bind"):
            with self.assertRaisesRegex(BackendUnavailableError, "missing pr_resident_shared_modes"):
                device._bind()
            lib.pr_resident_shared_modes_abi_version = Mock(return_value=1)
            with self.assertRaisesRegex(BackendUnavailableError, "creation export"):
                device._bind()
            lib.pr_resident_create_shared_ray_query = Mock(return_value=321)
            device._bind()
        with TemporaryDirectory() as directory:
            device.shader_directory = Path(directory)
            device.shader_sha256 = {}
            path = device.shader_directory/"resident_shared_instances.comp.spv"
            with self.assertRaisesRegex(BackendUnavailableError, "compute shader unavailable"):
                device._create(1,0,None,0,None,0)
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(BackendUnavailableError, "invalid shared TLAS"):
                device._create(1,0,None,0,None,0)
            # Dummy SPIR-V header exercises the Python byte/ABI handoff only.
            data = bytes.fromhex("03022307")+bytes(16)
            path.write_bytes(data)
            self.assertEqual(device._create(1,0,None,0,None,0), 321)
        args = lib.pr_resident_create_shared_ray_query.call_args.args
        self.assertEqual((bytes(args[-3]),bytes(args[-2]),args[-1]), (bytes(range(16)),data,len(data)))
        self.assertEqual(device.shader_sha256["shared_instances_compute"], sha256(data).hexdigest())
        lib.pr_resident_create_ray_query.assert_not_called()

    def test_camera_binding_requires_all_exports_and_64_bit_counter(self):
        from phymeo_rendering.backends._native_shared_matrices import bind_camera_capacity
        from phymeo_rendering.backends._native_device import Parameters

        lib = SimpleNamespace()
        with self.assertRaisesRegex(BackendUnavailableError, "missing pr_resident_shared_camera_abi_version"):
            bind_camera_capacity(lib)
        lib.pr_resident_shared_camera_abi_version = Mock(return_value=1)
        lib.pr_resident_set_shared_camera_capacity = Mock()
        lib.pr_resident_render_shared_camera_planes = Mock()
        with self.assertRaisesRegex(BackendUnavailableError, "missing pr_resident_shared_camera_d2d_bytes"):
            bind_camera_capacity(lib)
        lib.pr_resident_shared_camera_d2d_bytes = Mock()
        bind_camera_capacity(lib)
        self.assertEqual(lib.pr_resident_shared_camera_d2d_bytes.restype, ct.c_uint64)
        self.assertEqual(lib.pr_resident_render_shared_camera_planes.argtypes,
                         [ct.c_void_p, ct.c_uint32, ct.POINTER(Parameters), ct.c_uint32, ct.c_uint32,
                          ct.c_uint32, ct.POINTER(ct.c_uint8), ct.c_size_t])

    def test_reserved_cameras_extend_the_export_without_adding_draw_instances(self):
        from phymeo_rendering.backends._native_resident import ResidentDevice
        from phymeo_rendering.backends._native_device import NativeDevice
        from phymeo_rendering.backends._native_shared_matrices import SharedMatrices

        def base_init(owner, *args):
            owner._handle = 17
            owner._lib = SimpleNamespace(pr_resident_set_shared_camera_capacity=Mock(return_value=0))

        identity = (None,None,bytes(range(16)))
        with (patch.object(NativeDevice, "__init__", base_init),
              patch("phymeo_rendering.backends._native_shared_matrices.cuda_identity", return_value=identity)):
            device = ResidentDevice(shared_cuda_device=0, shared_camera_capacity=2)
            zero = ResidentDevice(shared_cuda_device=0)
        device._lib.pr_resident_set_shared_camera_capacity.assert_called_once_with(17,2)
        zero._lib.pr_resident_set_shared_camera_capacity.assert_not_called()
        device._instance_count = 3
        diag = SharedMatrices(device).diagnostics()
        self.assertEqual((diag["graphics_instances"],diag["camera_capacity"],diag["slot_bytes"]),
                         (3,2,5*128))
        self.assertEqual(device._instance_count, 3)
