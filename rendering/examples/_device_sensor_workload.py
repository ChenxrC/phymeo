"""One CUDA consumer shared by the CPU-upload and CUDA-delivery measurements."""
import numpy as np
import torch

from phymeo_rendering.outputs import Channel


DTYPES = {Channel.RGB: torch.uint8, Channel.FORWARD_DEPTH: torch.float32,
          Channel.INSTANCE_ID: torch.uint32, Channel.VALID: torch.uint8}
FEATURES = ('roi_red_sum', 'roi_green_sum', 'roi_blue_sum', 'visible_pixels',
            'background_positive_inf', 'moving_pixels', 'moving_depth_sum_mm', 'sensor_inconsistencies')


def shape(camera, channel):
    return (camera.height, camera.width, 3) if channel == Channel.RGB else (camera.height, camera.width)


class PinnedUploader:
    """Stage immutable Capture bytes into reusable pinned buffers and matching CUDA tensor slots."""

    def __init__(self, device):
        self.device = device
        self.slots = {}
        self.h2d_bytes = 0

    def upload(self, shots):
        results = []
        for shot in shots:
            planes = {}
            for plane in shot.planes:
                ch = plane.channel
                dims = shape(shot.camera, ch)
                slot = shot.frame_key.world_id, shot.camera.id, ch, dims
                if slot not in self.slots:
                    pin = torch.empty(dims, dtype=DTYPES[ch], device='cpu', pin_memory=True)
                    gpu = torch.empty(dims, dtype=DTYPES[ch], device=self.device)
                    self.slots[slot] = pin, pin.numpy(), gpu
                pin, host_view, gpu = self.slots[slot]
                np.copyto(host_view, np.frombuffer(plane.data, host_view.dtype).reshape(dims))
                gpu.copy_(pin, non_blocking=True)
                self.h2d_bytes += len(plane.data)
                planes[ch] = gpu
            results.append(planes)
        return results


def sensor_features(products):
    """Central-quarter RGB sums and moving-part range over the complete sensor image."""
    rows = []
    for planes in products:
        rgb, depth = planes[Channel.RGB], planes[Channel.FORWARD_DEPTH]
        ids, valid = planes[Channel.INSTANCE_ID].to(torch.int64), planes[Channel.VALID]
        h, w = depth.shape
        color = rgb[h//4:3*h//4, w//4:3*w//4].sum(dim=(0, 1), dtype=torch.int64)
        hit = valid == 1
        finite_hit = torch.isfinite(depth) & (depth > 0)
        background = torch.isposinf(depth)
        moving = hit & ((ids == 2) | (ids == 3)) & finite_hit
        selected = torch.where(moving, depth, 0.)
        bad = ((valid > 1) | (hit != ((ids > 0) & (ids <= 3))) | (hit != finite_hit)
               | ((~hit) & ((ids != 0) | ~background)))
        stats = torch.stack((hit.sum(), ((~hit) & background).sum(), moving.sum(),
                             (selected * 1000.).round().sum(dtype=torch.int64), bad.sum()))
        rows.append(torch.cat((color, stats)))
    return torch.stack(rows)


def numpy_features(shot):
    """Independent integer oracle from the CPU baseline's immutable native-raster buffers."""
    planes = {p.channel: p.data for p in shot.planes}
    h, w = shot.camera.height, shot.camera.width
    depth = np.frombuffer(planes[Channel.FORWARD_DEPTH], '<f4').reshape(h, w)
    ids = np.frombuffer(planes[Channel.INSTANCE_ID], '<u4').reshape(h, w)
    valid = np.frombuffer(planes[Channel.VALID], np.uint8).reshape(h, w)
    rgb = np.frombuffer(planes[Channel.RGB], np.uint8).reshape(h, w, 3)
    hit = valid == 1
    bg = np.isposinf(depth)
    bounded = (ids >= 1) & (ids <= 3)
    finite = np.isfinite(depth) & (depth > 0)
    moving = hit & ((ids == 2) | (ids == 3)) & finite
    bad = (valid > 1) | (hit != bounded) | (hit != finite) | ((~hit) & ((ids != 0) | ~bg))
    return [*rgb[h//4:3*h//4, w//4:3*w//4].sum(axis=(0, 1), dtype=np.int64).tolist(), int(hit.sum()),
            int((~hit & bg).sum()), int(moving.sum()), int(np.rint(depth[moving]*np.float32(1000)).sum(dtype=np.int64)),
            int(bad.sum())]


def audit_planes(shots, reference, device):
    """Explicit raw device-to-host oracle, called only outside performance measurements."""
    from hashlib import sha256

    if len(shots) != len(reference):
        raise ValueError('different numbers of camera exposures')
    results, total = [], 0
    for shot, old in zip(shots, reference, strict=True):
        if (shot.frame_key, shot.time_s, shot.camera) != (old.frame_key, old.time_s, old.camera):
            raise ValueError('device and host routes returned different exposures or view order')
        if tuple(shot.planes) != tuple(p.channel for p in old.planes):
            raise ValueError('device plane set or order differs from the requested planes')
        hashes = {}
        for p in old.planes:
            tensor = shot.planes[p.channel]
            if tensor.dtype != DTYPES[p.channel] or tensor.device != device or tuple(tensor.shape) != shape(shot.camera, p.channel):
                raise ValueError(f'invalid device, dtype or shape for {p.channel.value}')
            data = tensor.detach().cpu().numpy().tobytes(order='C')
            total += len(data)
            if data != p.data:
                raise ValueError(f'{shot.frame_key}/{shot.camera.id}/{p.channel.value}: device bytes differ')
            hashes[p.channel.value] = sha256(data).hexdigest()
        results.append(hashes)
    return results, total
