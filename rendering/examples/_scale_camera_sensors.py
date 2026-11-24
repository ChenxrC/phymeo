"""Full-channel CUDA consumer and independent NumPy reference for heterogeneous scenes."""
import numpy as np
import torch

from phymeo_rendering.device_images import DeviceImageBatch
from phymeo_rendering.outputs import Channel


def capture_worlds(route, sessions, workspace, packed, keys, request_factory, delivery_stream):
    """Return ordered CPU captures or a single waitable group from an already packed endpoint."""
    if route not in ('cpu_upload', 'direct_cuda'):
        raise ValueError('route must be cpu_upload or direct_cuda')
    results = []
    for key in keys:
        session = sessions[key]
        with session.publish_packed_device_poses(packed, workspace, key) as exposure:
            requests = tuple(request_factory(key, session, exposure))
            if route == 'cpu_upload':
                results.extend(session.capture_device_poses_batch(exposure, requests))
            else:
                results.append(session.capture_device_poses_device_batch(exposure, requests,
                                                                         consumer_stream=delivery_stream))
    return tuple(results) if route == 'cpu_upload' else DeviceImageBatch.combine(results)


class SceneSensorConsumer:
    """Apply the same eight sensor reductions to each ordered view on the caller's CUDA stream."""
    fields = ('central_red_sum', 'central_green_sum', 'central_blue_sum', 'visible_pixels',
              'positive_infinity_background_pixels', 'visible_id_sum', 'visible_depth_sum_mm', 'inconsistent_pixels')

    def __init__(self, scenes, keys, device, views_per_world=2):
        self.allowed = []
        self.device = torch.device(device)
        cache = {}
        for key in keys:
            ids = tuple(sorted(obj.id for obj in scenes[key].instances))
            if not ids:
                raise ValueError('the scale benchmark requires objects in every scene')
            if ids not in cache:
                dense = ids[0] == 1 and ids[-1] == len(ids)
                table = None if dense else torch.tensor(ids, device=self.device, dtype=torch.int64)
                cache[ids] = np.asarray(ids, np.uint32), table
            self.allowed.extend([cache[ids]] * views_per_world)
        self.ready = torch.cuda.current_stream(self.device).record_event()

    def __call__(self, products):
        torch.cuda.current_stream(self.device).wait_event(self.ready)
        features = []
        for planes, (allowed, table) in zip(products, self.allowed, strict=True):
            depth = planes[Channel.FORWARD_DEPTH]
            rgb, ids, valid = planes[Channel.RGB], planes[Channel.INSTANCE_ID].to(torch.int64), planes[Channel.VALID]
            h, w = depth.shape
            roi = rgb[h//4:3*h//4, w//4:3*w//4].sum(dim=(0, 1), dtype=torch.int64)
            hit, bg = valid == 1, torch.isposinf(depth)
            known = ((ids > 0) & (ids <= len(allowed))) if table is None else torch.isin(ids, table)
            finite = torch.isfinite(depth) & (depth > 0)
            good = hit & finite
            bad = ((valid > 1) | (hit != known) | (hit != finite) | ((~hit) & ((ids != 0) | ~bg)))
            z_mm = (torch.where(good, depth, 0.) * 1000.).round()
            stats = torch.stack((hit.sum(), ((~hit) & bg).sum(), torch.where(hit, ids, 0).sum(),
                                 z_mm.sum(dtype=torch.int64), bad.sum()))
            features.append(torch.cat((roi, stats)))
        return torch.stack(features)

    def numpy_oracle(self, captures):
        """Compute integer ground truth from public CPU planes, outside the timed consumer."""
        rows = []
        for capture, (allowed, _) in zip(captures, self.allowed, strict=True):
            raw = {p.channel: p.data for p in capture.planes}
            h, w = capture.camera.height, capture.camera.width
            ids = np.frombuffer(raw[Channel.INSTANCE_ID], '<u4').reshape(h, w)
            depth = np.frombuffer(raw[Channel.FORWARD_DEPTH], '<f4').reshape(h, w)
            valid = np.frombuffer(raw[Channel.VALID], 'u1').reshape(h, w)
            rgb = np.frombuffer(raw[Channel.RGB], 'u1').reshape(h, w, 3)
            hit, bg = valid == 1, np.isposinf(depth)
            finite = np.isfinite(depth) & (depth > 0)
            bad = (valid > 1) | (hit != np.isin(ids, allowed)) | (hit != finite) | (~hit & ((ids != 0) | ~bg))
            rows.append([*rgb[h//4:3*h//4, w//4:3*w//4].sum(axis=(0, 1), dtype=np.int64),
                         hit.sum(), ((~hit) & bg).sum(), ids[hit].sum(dtype=np.int64),
                         np.rint(depth[hit & finite]*np.float32(1000.)).sum(dtype=np.int64), bad.sum()])
        return np.asarray(rows, dtype=np.int64)
