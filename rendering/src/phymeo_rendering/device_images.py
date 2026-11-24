"""Owned Torch camera planes, cross-stream completion and explicit image readback."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from .cameras import Camera
from .outputs import Capture, Channel, ImagePlane
from .scene import FrameKey

if TYPE_CHECKING:
    import torch


def check_image_stream(stream, ordinal):
    import torch

    if (not isinstance(stream, torch.cuda.Stream)
            or stream.device != torch.device("cuda", ordinal)
            or stream == torch.cuda.default_stream(ordinal)):
        raise ValueError("device images require a non-default CUDA stream on the matched device")


@dataclass(frozen=True, slots=True)
class DeviceCapture:
    """One exposure. Each tensor owns a reference to the batch's Torch storage."""
    frame_key: FrameKey
    time_s: float
    camera: Camera
    planes: Mapping[Channel, torch.Tensor]


@dataclass(frozen=True, slots=True)
class _WorldImages:
    world: object
    captures: tuple[DeviceCapture, ...]
    event: torch.cuda.Event
    status: torch.Tensor | None


class DeviceImageBatch:
    """Waitable ordered results from the same committed physics observation."""

    def __init__(self, frame, ordinal, worlds):
        self._frame = frame
        self._ordinal = ordinal
        self._worlds = worlds

    @classmethod
    def combine(cls, batches):
        batches = tuple(batches)
        if not batches or any(not isinstance(batch, cls) for batch in batches):
            raise ValueError("provide at least one device image batch")
        first = batches[0]
        if any(b._frame is not first._frame or b._ordinal != first._ordinal for b in batches):
            raise ValueError("image batches must belong to the same physics frame and device")
        worlds = tuple(world for b in batches for world in b._worlds)
        if len({w.world for w in worlds}) != len(worlds):
            raise ValueError("duplicate world in combined device image batch")
        return cls(first._frame, first._ordinal, worlds)

    def _ready(self, stream):
        for world in self._worlds:
            if stream is None or world.status is not None:
                world.event.synchronize()
            if world.status is not None and not world.status.item():
                raise ValueError("invalid device image depth, instance ID or visibility agreement")
            if stream is not None:
                stream.wait_event(world.event)
                for capture in world.captures:
                    for tensor in capture.planes.values():
                        tensor.record_stream(stream)
        return tuple(c for world in self._worlds for c in world.captures)

    def wait(self, stream):
        """Order reads on stream; sensor batches also check their one-byte host verdict."""
        check_image_stream(stream, self._ordinal)
        return self._ready(stream)

    def synchronize(self):
        """Wait for all validation and image writes before exposing any world."""
        return self._ready(None)

    def readback_async(self, stream):
        """Queue tightly packed snapshots of all requested planes into pinned host memory."""
        import torch

        check_image_stream(stream, self._ordinal)
        rows = []
        with torch.cuda.stream(stream):
            for world in self._worlds:
                stream.wait_event(world.event)
                for capture in world.captures:
                    host_planes = []
                    for channel, tensor in capture.planes.items():
                        source = tensor.contiguous()
                        host = torch.empty(source.shape, dtype=source.dtype, pin_memory=True)
                        host.copy_(source, non_blocking=True)
                        tensor.record_stream(stream)
                        host_planes.append((channel, host))
                    rows.append((capture, host_planes))
            event = stream.record_event()
        return DeviceImageReadback(self, rows, event)


class DeviceImageReadback:
    def __init__(self, source, rows, event):
        self._source = source
        self._rows = rows
        self._event = event

    def result(self):
        self._event.synchronize()
        self._source.synchronize()
        return tuple(Capture(c.frame_key, c.time_s, c.camera,
                             tuple(ImagePlane(ch, plane.numpy().tobytes()) for ch, plane in planes))
                     for c, planes in self._rows)


def publish_device_images(active, views, results, instance_ids, stream):
    import torch

    if not isinstance(results, tuple) or len(results) != len(views):
        raise ValueError("device renderer returned the wrong number of views")
    checks = []
    captures = []
    types = {Channel.RGB: torch.uint8, Channel.FORWARD_DEPTH: torch.float32,
             Channel.INSTANCE_ID: torch.uint32, Channel.VALID: torch.uint8}
    dense_ids = not instance_ids or (min(instance_ids) == 1 and max(instance_ids) == len(instance_ids))
    allowed = None
    with torch.cuda.stream(stream), torch.inference_mode():
        for request, planes in zip(views, results, strict=True):
            if not isinstance(planes, Mapping) or tuple(planes) != request.channels:
                raise ValueError("device renderer returned different channels or order")
            camera = request.camera
            expected_visibility = None
            for channel, tensor in planes.items():
                shape = (camera.height, camera.width) + ((3,) if channel is Channel.RGB else ())
                if (not isinstance(tensor, torch.Tensor) or tensor.device != stream.device
                        or tensor.dtype != types[channel] or tensor.shape != shape):
                    raise ValueError(f"unexpected device image shape, dtype or device for {channel.value}")
                if channel is Channel.RGB:
                    continue
                if channel is Channel.FORWARD_DEPTH:
                    checks.append((tensor > 0).all())
                    visible = torch.isfinite(tensor)
                elif channel is Channel.INSTANCE_ID:
                    labels = tensor.view(torch.int32)
                    if dense_ids:
                        checks.append(((labels >= 0) & (labels <= len(instance_ids))).all())
                    else:
                        if allowed is None:
                            # Match the entire unsigned 32-bit domain using signed bit patterns.
                            values = [i if i < 2**31 else i - 2**32 for i in instance_ids]
                            allowed = torch.tensor([0, *values], dtype=torch.int32, device=stream.device)
                        checks.append(torch.isin(labels, allowed).all())
                    visible = labels != 0
                else:
                    checks.append((tensor <= 1).all())
                    visible = tensor != 0
                if expected_visibility is not None:
                    checks.append((visible == expected_visibility).all())
                else:
                    expected_visibility = visible
            captures.append(DeviceCapture(active.exposure.key, active.exposure.time_s, camera,
                                          MappingProxyType(dict(planes))))
        host_status = None
        if checks:
            status = torch.stack(checks).all().to(torch.uint8)
            host_status = torch.empty((), dtype=torch.uint8, pin_memory=True)
            host_status.copy_(status, non_blocking=True)
        event = stream.record_event()
    world = _WorldImages(active.binding.world_key, tuple(captures), event, host_status)
    return DeviceImageBatch(active.frame, stream.device.index, (world,))
