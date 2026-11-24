"""Public stream, publication and readback contracts on actual device tensors."""
import os
from types import SimpleNamespace
import unittest

from phymeo_rendering.cameras import Camera
from phymeo_rendering.device_images import DeviceImageBatch, publish_device_images
from phymeo_rendering.outputs import Channel
from phymeo_rendering.scene import FrameKey, Pose


@unittest.skipUnless(os.environ.get("PHYMEO_CUDA_DEVICE_IMAGES") == "1", "requires an opted-in CUDA device")
class DeviceImageOutputTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.writer = torch.cuda.Stream()
        self.reader = torch.cuda.Stream()
        self.frame = object()

    def submit(self, world, *, invalid=None, frame=None):
        torch = self.torch
        key = FrameKey("test-scene", str(world), 0, 1, 0, "accepted_endpoint")
        camera = Camera("eye", key, Pose((0., 0., 0.), (1., 0., 0., 0.)), 2, 1, 2., 2., 1., .5, .01, 5.)
        active = SimpleNamespace(frame=self.frame if frame is None else frame, binding=SimpleNamespace(world_key=world),
                                 exposure=SimpleNamespace(key=key, time_s=.25))
        with torch.cuda.stream(self.writer):
            rgba = torch.full((1, 2, 4), world, dtype=torch.uint8, device=self.writer.device)
            depth = torch.tensor([[.5, float("inf")]], device=self.writer.device)
            identity = torch.tensor([[-1, 0]], dtype=torch.int32, device=self.writer.device)
            valid = torch.tensor([[1, 0]], dtype=torch.uint8, device=self.writer.device)
            if invalid == "depth":
                depth.fill_(float("nan"))
            if invalid == "id":
                identity.fill_(42)
            if invalid == "valid":
                valid.fill_(2)
            if invalid == "agreement":
                valid.zero_()
            planes = dict(zip(Channel, (rgba[..., :3], depth, identity.view(torch.uint32), valid), strict=True))
        request = SimpleNamespace(camera=camera, channels=tuple(Channel))
        return publish_device_images(active, (request,), (planes,), {2**32-1}, self.writer)

    def test_reordered_worlds_async_host_and_high_ids(self):
        group = DeviceImageBatch.combine([self.submit(2), self.submit(1)])
        copies = group.readback_async(self.reader)
        captures = group.wait(self.reader)
        self.assertEqual([c.frame_key for c in copies.result()], [c.frame_key for c in captures])
        for world, capture in zip((2, 1), copies.result(), strict=True):
            self.assertEqual(capture.planes[0].data, bytes([world])*6)
            self.assertEqual(capture.planes[2].data, b"\xff"*4 + bytes(4))
        self.assertEqual(captures[0].planes[Channel.RGB].stride()[-2:], (4, 1))

    def test_invalid_last_world_never_publishes_partial_group(self):
        good = self.submit(1)
        for fault in ("depth", "valid", "id", "agreement"):
            with self.subTest(fault=fault):
                group = DeviceImageBatch.combine((good, self.submit(2, invalid=fault)))
                with self.assertRaisesRegex(ValueError, "invalid device image"):
                    group.wait(self.reader)
                with self.assertRaisesRegex(ValueError, "invalid device image"):
                    group.readback_async(self.reader).result()

    def test_reject_wrong_stream_and_incompatible_worlds(self):
        batch = self.submit(1)
        with self.assertRaisesRegex(ValueError, "non-default"):
            batch.wait(self.torch.cuda.default_stream())
        with self.assertRaisesRegex(ValueError, "duplicate world"):
            DeviceImageBatch.combine((batch, batch))
        with self.assertRaisesRegex(ValueError, "same physics frame"):
            DeviceImageBatch.combine((batch, self.submit(2, frame=object())))
