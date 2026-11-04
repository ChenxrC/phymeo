"""Opt-in TYX first hits for analytic and triangulated collision surfaces."""

from numbers import Integral

import numpy as np

from ..geometry.raycast import RayHits, _frozen, _ray_batch, _ray_shape, _surfaces
from .tyx_math import TyxMath, as_compute

_MAX_PATCHES = 2**16
_MAX_TESTS = 2**26
_F32 = np.finfo(np.float32)
_CURVES = {"cylinder": 5, "cone": 6, "ellipsoid": 7}


def _unit(value):
    # Scale before taking a norm, including normals registered with non-unit magnitude.
    scaled = value / np.max(np.abs(value))
    return scaled / np.hypot.reduce(scaled)


def _patches(model, poses, world, reference):
    for index, kind, points, param in _surfaces(model, poses, world):
        p = points - reference
        if not np.isfinite(p).all():
            raise ValueError("ray scene exceeds the FP32 coordinate domain after rebasing")
        if kind in _CURVES:
            frame, dimensions = param
            yield _CURVES[kind], index, p[None, :], frame.T, dimensions
        elif kind in ("triangle", "plane"):
            yield (4 if kind == "triangle" else 3), index, p, _unit(param), (0.0, 0.0)
        elif kind != "round":
            raise NotImplementedError(f"unsupported TYX ray geometry: {kind}")
        elif len(p) == 1 or np.array_equal(points[0], points[1]):
            yield 0, index, p[:1], (0.0, 0.0, 0.0), (param, 0.0)
        else:
            axis = points[1] - points[0]
            length = np.hypot.reduce(axis)
            if not np.isfinite(length) or not length:
                raise ValueError("capsule length is outside the FP32 ray domain")
            axis = axis / length
            yield 1, index, p[:1], -axis, (param, 0.0)
            yield 1, index, p[1:], axis, (param, 0.0)
            yield 2, index, p[:1], axis, (param, length)


def _geometry(model, poses, origin, ray_count, world, capacity):
    reference = origin if origin.ndim == 1 else (origin[0] if len(origin) else np.zeros(3))
    local = origin - reference
    extent = float(np.max(np.abs(local), initial=0.0))
    patches = []
    for record in _patches(model, poses, world, reference):
        if len(patches) >= _MAX_PATCHES or (len(patches) + 1) * ray_count > capacity:
            raise ValueError(
                f"TYX ray capacity exceeded at {len(patches) + 1} patches × {ray_count} rays"
            )
        patches.append(record)
        _, _, points, _, size = record
        extent = max(extent, float(np.max(np.abs(points))), *size)
    if not patches or not ray_count:
        return None
    if not np.isfinite(extent):
        raise ValueError("ray scene exceeds the FP32 coordinate domain after rebasing")
    # One exact binary scale and translation preserve angles and metric conversion. Vertices
    # remain in a modest numeric range even if the scene is far from world zero.
    scale = float(np.ldexp(1.0, int(np.frexp(extent)[1]) - 1)) if extent else 1.0
    tags = np.empty((len(patches), 2), np.int32)
    width = 15 if any(record[0] == 7 for record in patches) else 14
    data = np.zeros((len(patches), width))
    for row, (kind, index, points, axis, size) in enumerate(patches):
        p = points / scale
        tags[row] = kind, index
        data[row, :3] = p[0]
        metric = np.asarray(size) / scale
        data[row, 12 : 12 + len(size)] = metric
        if kind >= 5:
            if np.min(metric) < 4 * _F32.eps:
                raise ValueError(
                    "analytic ray surface is unresolved at the scene's FP32 resolution"
                )
            data[row, 3:12] = axis.reshape(9)
            continue
        data[row, 3:6] = axis
        # Avoid returning finite-looking ranges when an entire transverse feature is smaller
        # than the FP32 precision available in this query envelope.
        if kind <= 2:
            if metric[0] < 4 * _F32.eps or (kind == 2 and metric[1] < 4 * _F32.eps):
                raise ValueError("round ray surface is unresolved at the scene's FP32 resolution")
        if kind == 4:
            data[row, 6:12] = p[1:].reshape(6)
            e1, e2 = p[1] - p[0], p[2] - p[0]
            longest = max(np.hypot.reduce(e) for e in (e1, e2, e2 - e1))
            if not longest or np.hypot.reduce(np.cross(e1, e2)) < longest * (4 * _F32.eps):
                raise ValueError("triangle altitude is unresolved at the scene's FP32 resolution")
    return tags, as_compute(data, "FP32 ray surfaces"), local / scale, scale


def _clip(near, far, scale):
    # Direct each rounded boundary inwards. Every reported metric distance must be in the
    # caller's original closed interval, including when one boundary lies between FP32 values.
    near = near / scale
    far = min(far / scale, float(_F32.max))
    if near > _F32.max or (0 < near < _F32.tiny) or far < _F32.tiny:
        raise ValueError("ray clipping interval exceeds the supported FP32 range")
    low, high = np.float32(near), np.float32(far)
    if float(low) < near:
        low = np.nextafter(low, np.float32(np.inf))
    if float(high) > far:
        high = np.nextafter(high, np.float32(0))
    if low > high:
        raise ValueError("ray clipping interval contains no representable FP32 distance")
    return float(low), float(high)


def _world_packet(model, poses, origin, rays, world, near, far, capacity):
    geom = _geometry(model, poses, origin, len(rays), world, capacity)
    if geom is None:
        return None
    tags, data, local, scale = geom
    close, distant = _clip(near, far, scale)
    beams = as_compute(np.column_stack((np.broadcast_to(local, rays.shape), rays)), "FP32 beams")
    return beams, tags, data, scale, close, distant


def _routed_packet(model, poses, origin, rays, groups, near, far, capacity):
    if not groups:
        return None
    tag_blocks, surfaces = [], []
    # Empty-world directions remain valid even though their empty ranges perform no tests.
    beams = np.zeros((len(rays), 8), np.float32)
    beams[:, 3:6] = rays
    bounds = np.zeros((len(rays), 2), np.int32)
    scales = np.ones(len(rays))
    total = tests = 0
    for key, slots in groups:
        eye = origin if origin.ndim == 1 else origin[slots]
        packet = _world_packet(model, poses, eye, rays[slots], key, near, far, capacity - tests)
        if packet is None:
            continue
        local_beams, tags, data, scale, low, high = packet
        end = total + len(data)
        if end > _MAX_PATCHES:
            raise ValueError("TYX ray capacity exceeded by selected worlds' surface tables")
        bounds[slots] = (total, end)
        beams[slots, :6] = local_beams
        beams[slots, 6:] = (low, high)
        scales[slots] = scale
        tag_blocks.append(tags)
        surfaces.append(data)
        total = end
        tests += len(slots) * len(data)
    if not total:
        return None
    width = max(s.shape[1] for s in surfaces)
    packed = np.zeros((total, width), np.float32)
    start = 0
    for block in surfaces:
        packed[start : start + len(block), : block.shape[1]] = block
        start += len(block)
    # Rows [0,total) identify original shapes; the i32 tail gives each beam its own [start,end).
    return beams, np.vstack((*tag_blocks, bounds)), packed, scales, 0.0, 0.0


class TyxRaycaster:
    """Metric first hits from selected fixed worlds, using an explicit FP32 TYX mode.

    Each call consumes its supplied pose and ray origins. The host only prepares registered
    geometry and units. The kernel evaluates every ray against every selected patch, including
    both hemisphere roots and the finite shaft of a capsule. Selected unsupported surfaces
    and over-capacity queries raise before launch. Use as a context manager to close the runtime.
    """

    def __init__(self, mode="cuda", *, cache_root=None, max_ray_patch_tests=_MAX_TESTS):
        if mode not in ("cuda", "interpret"):
            raise ValueError("ray mode must be cuda or interpret")
        if (
            isinstance(max_ray_patch_tests, bool)
            or not isinstance(max_ray_patch_tests, Integral)
            or not 1 <= max_ray_patch_tests <= _MAX_TESTS
        ):
            raise ValueError(f"ray test capacity must be an integer from 1 to {_MAX_TESTS}")
        self.mode = mode
        self._cache = cache_root
        self._capacity = int(max_ray_patch_tests)
        self._math = None
        self._closed = False
        self._uploaded = self._downloaded = 0

    def __enter__(self):
        if self._closed:
            raise RuntimeError("TYX raycaster is closed")
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._math is not None and not self._closed:
            self._math.close()
        self._closed = True

    def stats(self):
        counts = (
            self._math.stats()
            if self._math is not None
            else dict(
                mode=self.mode,
                compute_dtype="float32",
                launches=0,
                completed_launches=0,
                uploads=0,
                downloads=0,
            )
        )
        return dict(counts, uploaded_bytes=self._uploaded, downloaded_bytes=self._downloaded)

    def query(self, model, poses, origins_m, directions, *, world=0, near_m=0.0, far_m=np.inf):
        """First hit per unit ray, including either endpoint of the metric clip interval."""
        if self._closed:
            raise RuntimeError("TYX raycaster is closed")
        shape, origin, rays, near, far, groups = _ray_batch(
            model, poses, origins_m, directions, world, near_m, far_m
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            if groups is None:
                packet = _world_packet(model, poses, origin, rays, world, near, far, self._capacity)
            else:
                packet = _routed_packet(
                    model, poses, origin, rays, groups, near, far, self._capacity
                )
        keys = tuple(s.key for s in model.shapes)
        if packet is None:
            return RayHits(
                _frozen(np.full(shape, np.inf)),
                _frozen(np.full(shape, -1, dtype=np.int32)),
                _frozen(np.zeros((*shape, 3))),
                keys,
            )
        beams, tags, data, scale, close, distant = packet
        labels = np.zeros(len(rays), np.int32)
        hits = np.zeros((len(rays), 4), np.float32)
        if self._math is None:
            self._math = TyxMath(self.mode, cache_root=self._cache)

        from ._tyx_raycast_kernels import first_hits

        arrays = (beams, tags, data, labels, hits)
        before = self._math.stats()
        try:
            labels, hits = self._math.call(
                first_hits,
                len(rays),
                len(data),
                int(groups is not None),
                close,
                distant,
                *arrays,
                outputs=(8, 9),
            )
        finally:
            after = self._math.stats()
            self._uploaded += sum(x.nbytes for x in arrays[: after["uploads"] - before["uploads"]])
            self._downloaded += sum(
                x.nbytes for x in arrays[3 : 3 + after["downloads"] - before["downloads"]]
            )
        if np.any(labels == -2):
            raise FloatingPointError(
                f"unresolved FP32 ray arithmetic at row {np.flatnonzero(labels == -2)[0]}"
            )
        with np.errstate(over="raise", invalid="raise"):
            distances = hits[:, 0].astype(float) * scale
        distances[labels == -1] = np.inf
        return _ray_shape(
            RayHits(_frozen(distances), _frozen(labels), _frozen(hits[:, 1:].astype(float)), keys),
            shape,
        )
