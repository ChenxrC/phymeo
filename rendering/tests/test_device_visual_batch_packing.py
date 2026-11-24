"""GPU layout and error propagation controls with known rigid-body transforms."""
import os
from types import SimpleNamespace
import unittest

from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.device_pose_packing import DeviceCameraMount, DeviceVisualBatchBinding, DeviceVisualBinding
from phymeo_rendering.scene import Instance, Pose, VisualScene


@unittest.skipUnless(os.environ.get('PHYMEO_CUDA_POSE_PACKING') == '1', 'explicit CUDA packing test slot')
class BatchLayouts(unittest.TestCase):
    def setUp(self):
        import torch

        self.stream = torch.cuda.Stream(device=0)
        source = dict(mode='cuda', device=0, endpoint_body_kinematics=True, world_keys=(41, 7),
                      source=dict(worlds=((41, 'short', 0, 0), (7, 'long', 0, 0))))
        ws = SimpleNamespace(binding=source, model_metadata=dict(short=dict(body_names=('world', 'arm')),
                                                                  long=dict(body_names=('world', 'arm', 'eye'))))
        mesh = (Mesh('box', 'shape.obj'),)
        paint = (Material('paint', (.3, .4, .5), 0., .5),)
        first = Instance(1, 'box', 'paint', 'arm')
        small = VisualScene('small', mesh, paint, (first,))
        large = VisualScene('large', mesh, paint, (first, Instance(2, 'box', 'paint', 'other')))
        self.short = DeviceVisualBinding(small, ws, 41, {'arm': 1})
        self.long = DeviceVisualBinding(large, ws, 7, {'arm': 1, 'other': 2},
                                        camera_mounts=(DeviceCameraMount('wrist', 2, Pose((.25, 0., 0.))),))
        with torch.cuda.stream(self.stream):
            rows = torch.eye(4, device='cuda:0', dtype=torch.float32)[:3].repeat(5, 1, 1)
            rows[:, 0, 3] = torch.arange(5, device='cuda:0')
        self.snapshot = SimpleNamespace(frame=SimpleNamespace(binding=source), body_offsets=(0, 2, 5),
                                        transforms=rows)

    def test_reordered_models_and_unmounted_subset(self):
        layout = DeviceVisualBatchBinding((self.long, self.short))
        published = layout.pack(self.snapshot, producer_stream=self.stream)
        for key, expected in ((7, [3., 4., 4.25]), (41, [1.])):
            _, records, metadata = published._view(key)
            self.assertEqual(records[:, 0, 3, 0].tolist(), expected)
            self.assertEqual(records[:, 1, 3, 0].tolist(), [-x for x in expected])
            self.assertEqual(metadata, {0: Pose((4.25, 0., 0.))} if key == 7 else {})
        self.assertEqual((layout.readback_calls, layout.status_readback_bytes,
                          layout.camera_metadata_readback_bytes, layout.packed_output_bytes), (1, 1, 48, 512))
        with self.assertRaisesRegex(ValueError, 'not in this'):
            published._view(500)
        only = DeviceVisualBatchBinding((self.short,))
        self.assertEqual(only.pack(self.snapshot, producer_stream=self.stream)._view(41)[2], {})
        self.assertEqual((only.readback_calls, only.camera_metadata_readback_bytes), (1, 0))

    def test_invalid_last_world_rejects_entire_pack(self):
        import torch

        with self.assertRaisesRegex(ValueError, 'duplicate'):
            DeviceVisualBatchBinding((self.short, self.short))
        layout = DeviceVisualBatchBinding((self.long, self.short))
        with self.assertRaisesRegex(ValueError, 'non-default'):
            layout.pack(self.snapshot, producer_stream=torch.cuda.default_stream(0))
        with torch.cuda.stream(self.stream):
            self.snapshot.transforms[1, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'reflected'):
            layout.pack(self.snapshot, producer_stream=self.stream)
        self.assertEqual((layout.readback_calls, layout.packs, layout.packed_output_bytes), (1, 0, 0))
