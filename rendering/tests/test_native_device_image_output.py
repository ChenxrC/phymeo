"""Optional ABI layout, packed tensor views, and CUDA attachment ownership."""

from contextlib import nullcontext
import ctypes as ct
from importlib.util import find_spec
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from phymeo_rendering.backends import BackendUnavailableError
from phymeo_rendering.backends._native_camera_batch import batch_views
from phymeo_rendering.backends._native_device import Parameters
from phymeo_rendering.backends._native_device_output import (
    DeviceImageOutput, _Handoff, _OutputInfo, _plane_views, bind_device_output,
)
from phymeo_rendering.backends._native_resident import ResidentDevice
from phymeo_rendering.backends._native_shared_matrices import _Export
from phymeo_rendering.outputs import Channel


def row(width, channels):
    params = Parameters()
    params.intrinsics[:2] = (1, 1)
    params.clipping[:] = (width, 1, .1, 10)
    params.options[1] = .001
    params.counts[2] = int(Channel.RGB in channels)
    return params, width, 1, channels, None


class DeviceOutputLayoutTests(unittest.TestCase):
    def test_optional_abi_rejects_old_or_partial_libraries(self):
        with self.assertRaisesRegex(BackendUnavailableError, 'missing pr_resident_device_output_abi_version'):
            bind_device_output(SimpleNamespace())
        with self.assertRaisesRegex(BackendUnavailableError, 'ABI 2'):
            bind_device_output(SimpleNamespace(pr_resident_device_output_abi_version=Mock(return_value=2)))
        with self.assertRaisesRegex(BackendUnavailableError, 'missing pr_resident_device_output_export'):
            bind_device_output(SimpleNamespace(pr_resident_device_output_abi_version=Mock(return_value=1)))
        self.assertEqual(ct.sizeof(_OutputInfo), 40)

    @unittest.skipUnless(find_spec('torch'), 'Torch tensor views are optional')
    def test_odd_extent_alignment_and_owned_typed_views(self):
        import torch
        device = ResidentDevice.__new__(ResidentDevice)
        device._handle, device._uploaded, device._instance_count = 7, True, 0
        device._last_info = dict(max_width=4096, max_height=4096)
        device._render_planes_function = None
        rows = (row(1, (Channel.VALID,)), row(2, (Channel.VALID, Channel.INSTANCE_ID,
                                                   Channel.RGB, Channel.FORWARD_DEPTH)))
        views, spans, size = batch_views(device, rows, device_output=True)
        self.assertEqual([(s[0], s[1]) for s in spans], [(0, 1), (4, 30)])
        raw = (bytes([1, 0xcd, 0xcd, 0xcd, 1, 2, 3, 255, 4, 5, 6, 255])
               + struct.pack('<ffII', 1.5, float('inf'), 0xf0000001, 0) + bytes([1, 0]))
        self.assertEqual(len(raw), size)
        slab = torch.tensor(list(raw), dtype=torch.uint8)
        first, second = _plane_views(torch, slab, views, spans)
        self.assertEqual(list(second), list(rows[1][3]))
        self.assertEqual(first[Channel.VALID].tolist(), [[1]])
        self.assertEqual(second[Channel.RGB].tolist(), [[[1, 2, 3], [4, 5, 6]]])
        self.assertEqual(second[Channel.RGB].stride(), (8, 4, 1))
        self.assertEqual(second[Channel.INSTANCE_ID].dtype, torch.uint32)
        self.assertEqual(second[Channel.INSTANCE_ID].tolist(), [[0xf0000001, 0]])
        self.assertEqual(second[Channel.FORWARD_DEPTH].tolist(), [[1.5, float('inf')]])
        self.assertEqual(second[Channel.VALID].tolist(), [[1, 0]])
        slab[4] = 99
        del slab
        self.assertEqual(second[Channel.RGB][0, 0, 0].item(), 99)


class DeviceOutputOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.actions = []
        self.phase = 1

        def cuda(action):
            def call(*args):
                self.actions.append(action)
                return (0,)
            return call

        driver = SimpleNamespace(CUstream=int, cuWaitExternalSemaphoresAsync=cuda('wait ready'),
                                 cuSignalExternalSemaphoresAsync=cuda('signal writable'),
                                 CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS=object,
                                 CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS=object)
        lib = Mock(pr_resident_device_output_abi_version=Mock(return_value=1))
        exchange = SimpleNamespace(_call=self.native, _current=nullcontext)
        self.output = DeviceImageOutput(SimpleNamespace(_lib=lib, _shared_identity=(driver, 0, bytes(16)),
                                                       _handle=7, _shared_exchange=exchange))
        self.output._ready, self.output._writable = 9, 10

    def native(self, symbol, *args):
        self.actions.append(symbol)
        if symbol.endswith('_info'):
            result = ct.cast(args[0], ct.POINTER(_OutputInfo)).contents
            result.abi_version, result.phase = 1, self.phase
        if symbol.endswith('_export'):
            capacity, pointer = args
            result = ct.cast(pointer, ct.POINTER(_Export)).contents
            result.memory, result.writable, result.ready = 1, 2, 3
            result.buffer_bytes, result.allocation_bytes = capacity, 4096

    def handoff(self, released):
        event = SimpleNamespace(record=lambda stream: self.actions.append('record event'),
                                synchronize=lambda: self.actions.append('wait event'))
        pending = _Handoff(SimpleNamespace(cuda_stream=41), event, object(), released)
        self.output._handoff = pending
        return pending

    def test_error_before_submit_does_not_wait_on_unsignaled_semaphore(self):
        self.handoff(None)
        self.output._finish()
        self.assertEqual(self.actions, ['pr_resident_device_output_info'])
        self.assertIsNone(self.output._handoff)

    def test_error_after_submit_discards_and_returns_ready_slot(self):
        self.phase = 2
        self.handoff(None)
        self.output._finish()
        self.assertEqual(self.actions, ['pr_resident_device_output_info', 'wait ready', 'signal writable',
                                       'record event', 'wait event', 'pr_resident_device_output_return'])
        self.assertIsNone(self.output._handoff)

    def test_unfinished_vulkan_fence_retries_without_queueing_cuda_wait(self):
        self.phase = 3
        handoff = self.handoff(None)
        with self.assertRaisesRegex(TimeoutError, 'fence still pending'):
            self.output.close()
        self.assertEqual(self.actions, ['pr_resident_device_output_info'])
        self.assertIs(self.output._handoff, handoff)
        self.phase = 2
        self.output.close()
        self.assertIsNone(self.output._handoff)
        self.assertEqual(self.actions.count('wait ready'), 1)

    def test_release_retry_waits_without_signalling_twice(self):
        handoff = self.handoff(True)
        handoff.ready_wait_queued = handoff.writable_signal_queued = handoff.recorded = True
        self.output._call = Mock(side_effect=[TimeoutError('pending'), None])
        with self.assertRaises(TimeoutError):
            self.output.close()
        self.assertIs(self.output._handoff, handoff)
        self.output.close()
        self.assertEqual(self.actions, ['wait event', 'wait event'])
        self.assertIsNone(self.output._handoff)

    def test_import_rollback_and_grow_only_reuse(self):
        def mapping(driver, exported, imports):
            def destroy(index):
                self.actions.append(f'destroy {index}')
                return (0,)
            for index in range(4):
                imports.append((destroy, index))
            return 11, 12, 13

        def failing_import(driver, exported, imports):
            mapping(driver, exported, imports)
            raise RuntimeError('bad external semaphore')

        name = 'phymeo_rendering.backends._native_device_output.import_external_slot'
        with patch(name, failing_import), self.assertRaisesRegex(RuntimeError, 'bad external semaphore'):
            self.output._reserve(4)
        self.assertEqual(self.actions[-6:], ['destroy 3', 'destroy 2', 'destroy 1', 'destroy 0',
                                            'pr_resident_device_output_drop_exports', 'pr_resident_device_output_retire'])
        with patch(name, mapping):
            self.output._reserve(4)
            self.output._reserve(3)
            self.output._reserve(5)
            self.output.close()
        self.assertEqual((self.output._created, self.output._destroyed, self.output._generations), (12, 12, 3))
        self.assertEqual((self.output._capacity, len(self.output._imports)), (0, 0))
