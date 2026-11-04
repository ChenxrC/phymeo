"""Opt-in TYX first hits for sampled worlds with round, flat and triangulated surfaces."""

from numbers import Integral

import numpy as np

from ..geometry.raycast import RayHits, _frozen, _ray_inputs, _surfaces
from .tyx_math import TyxMath, as_compute

_MAX_PATCHES = 2**16
_MAX_TESTS = 2**26
_F32 = np.finfo(np.float32)


def _unit(value):
    # Scale before taking a norm, including normals registered with non-unit magnitude.
    scaled = value / np.max(np.abs(value))
    return scaled / np.hypot.reduce(scaled)


def _patches(model, poses, world, reference):
    for index, kind, points, param in _surfaces(model, poses, world):
        if kind not in ("round", "triangle", "plane"):
            raise NotImplementedError(f"unsupported TYX ray geometry: {kind}")
        p = points - reference
        if not np.isfinite(p).all():
            raise ValueError("ray scene exceeds the FP32 coordinate domain after rebasing")
        if kind != "round":
            yield (4 if kind == "triangle" else 3), index, p, _unit(param), 0.0, 0.0
        elif len(p) == 1 or np.array_equal(points[0], points[1]):
            yield 0, index, p[:1], (0.0, 0.0, 0.0), param, 0.0
        else:
            axis = points[1] - points[0]
            length = np.hypot.reduce(axis)
            if not np.isfinite(length) or not length:
                raise ValueError("capsule length is outside the FP32 ray domain")
            axis = axis / length
            yield 1, index, p[:1], -axis, param, 0.0
            yield 1, index, p[1:], axis, param, 0.0
            yield 2, index, p[:1], axis, param, length


def _geometry(model, poses, origin, ray_count, world, capacity):
    reference = origin if origin.ndim == 1 else (origin[0] if len(origin) else np.zeros(3))
    local = origin - reference
    extent = float(np.max(np.abs(local), initial=0.0))
    patches = []
    for record in _patches(model, poses, world, reference):
        if len(patches) >= _MAX_PATCHES or (len(patches) + 1) * ray_count > capacity:
            raise ValueError(f"TYX ray capacity exceeded at {len(patches) + 1} patches × {ray_count} rays")
        patches.append(record)
        _, _, points, _, radius, length = record
        extent = max(extent, float(np.max(np.abs(points))), radius, length)
    if not patches or not ray_count:
        return None
    if not np.isfinite(extent):
        raise ValueError("ray scene exceeds the FP32 coordinate domain after rebasing")
    # One exact binary scale and translation preserve angles and metric conversion. Vertices
    # remain in a modest numeric range even if the scene is far from world zero.
    scale = float(np.ldexp(1.0, int(np.frexp(extent)[1]) - 1)) if extent else 1.0
    tags = np.empty((len(patches), 2), np.int32)
    data = np.zeros((len(patches), 14))
    for row, (kind, index, points, axis, radius, length) in enumerate(patches):
        p = points / scale
        tags[row] = kind, index
        data[row, :3] = p[0]
        data[row, 3:6] = axis
        data[row, 12:] = radius / scale, length / scale
        # Avoid returning finite-looking ranges when an entire transverse feature is smaller
        # than the FP32 precision available in this query envelope.
        if kind <= 2:
            if radius / scale < 4 * _F32.eps or (kind == 2 and length / scale < 4 * _F32.eps):
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


class TyxRaycaster:
    """Metric first hits from one fixed world, using an explicit FP32 TYX execution mode.

    Each call consumes its supplied pose and ray origins. The host only prepares registered
    geometry and units. The kernel evaluates every ray against every selected patch, including
    both hemisphere roots and the finite shaft of a capsule. Selected unsupported surfaces
    and over-capacity queries raise before launch. Use as a context manager to close the runtime.
    """

    def __init__(self, mode="cuda", *, cache_root=None, max_ray_patch_tests=_MAX_TESTS):
        if mode not in ("cuda", "interpret"):
            raise ValueError("ray mode must be cuda or interpret")
        if (isinstance(max_ray_patch_tests, bool) or not isinstance(max_ray_patch_tests, Integral)
            or not 1 <= max_ray_patch_tests <= _MAX_TESTS):
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
            self._math.stats() if self._math is not None else
            dict(mode=self.mode, compute_dtype="float32", launches=0, completed_launches=0, uploads=0, downloads=0)
        )
        return dict(counts, uploaded_bytes=self._uploaded, downloaded_bytes=self._downloaded)

    def query(self, model, poses, origins_m, directions, *, world=0, near_m=0.0, far_m=np.inf):
        """First hit per unit ray, including either endpoint of the metric clip interval."""
        if self._closed:
            raise RuntimeError("TYX raycaster is closed")
        origin, rays, near, far = _ray_inputs(origins_m, directions, near_m, far_m)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            geometry = _geometry(model, poses, origin, len(rays), world, self._capacity)
        keys = tuple(s.key for s in model.shapes)
        if geometry is None:
            return RayHits(_frozen(np.full(len(rays), np.inf)),
                           _frozen(np.full(len(rays), -1, dtype=np.int32)),
                           _frozen(np.zeros((len(rays), 3))), keys)
        tags, data, local, scale = geometry
        close, distant = _clip(near, far, scale)
        beams = as_compute(np.column_stack((np.broadcast_to(local, rays.shape), rays)), "FP32 beams")
        labels = np.zeros(len(rays), np.int32)
        hits = np.zeros((len(rays), 4), np.float32)
        if self._math is None:
            self._math = TyxMath(self.mode, cache_root=self._cache)

        from ._tyx_raycast_kernels import first_hits

        arrays = (beams, tags, data, labels, hits)
        before = self._math.stats()
        try:
            labels, hits = self._math.call(
                first_hits, len(rays), len(tags), close, distant, *arrays, outputs=(7, 8)
            )
        finally:
            after = self._math.stats()
            self._uploaded += sum(x.nbytes for x in arrays[:after["uploads"] - before["uploads"]])
            self._downloaded += sum(x.nbytes for x in arrays[3:3 + after["downloads"] - before["downloads"]])
        if np.any(labels == -2):
            raise FloatingPointError(f"unresolved FP32 ray arithmetic at row {np.flatnonzero(labels == -2)[0]}")
        with np.errstate(over="raise", invalid="raise"):
            distances = hits[:, 0].astype(float) * scale
        distances[labels == -1] = np.inf
        return RayHits(_frozen(distances), _frozen(labels), _frozen(hits[:, 1:].astype(float)), keys)
