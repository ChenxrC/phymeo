"""Explicit bounded TYX segment--triangle packets; HOST only packs/checks data.

No collision response or closed-solid classification. A packet describes the
supplied candidate membership, not a certificate that a broad phase was complete.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType

import numpy as np

from .mujoco_tyx._resources import DeviceResources

_U = 2.0**-24
_CAPS = dict(vertex_capacity=512, face_capacity=128, segment_capacity=64, pair_capacity=8192)
_ARRAYS = (
    "vertex_ids",
    "facet_ids",
    "faces",
    "vertices",
    "segment_ids",
    "segments",
    "radii",
    "candidates",
    "origin_m",
)


def _immutable(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _mapping(values):
    return MappingProxyType({name: _immutable(value) for name, value in values.items()})


def _array(value, dtype, shape, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise ValueError(f"{name} requires exact {np.dtype(dtype)} array")
    if value.ndim != len(shape) or any(
        n is not None and n != a for n, a in zip(shape, value.shape, strict=True)
    ):
        raise ValueError(f"invalid {name} shape")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"nonfinite {name}")
    return _immutable(value)


def _ids(value, name):
    result = _array(value, np.int64, (None,), name)
    if np.any(result < 0) or len(set(map(int, result))) != len(result):
        raise ValueError(f"invalid or duplicate {name}")
    return result


def _binding(metadata, arrays):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for key in _ARRAYS:
        value = arrays[key]
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class _RuntimeLease:
    """Keep the actual runtime alive while a query owns dependent buffers."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.closed = False

    def close(self):
        # Query cleanup releases this lease, never the batch/caller runtime.
        self.closed = True


class _BufferLease:
    """Only a normally returned opaque free establishes released byte count."""

    def __init__(self, array, is_cuda, size, counts):
        self.array, self.is_cuda, self.size, self.counts = array, is_cuda, size, counts
        self.released = False

    def free(self):
        if self.released:
            return
        if self.is_cuda:
            self.array.free()
        self.released = True
        self.counts["live_bytes"] -= self.size


@dataclass(frozen=True)
class TriangleDistanceRow:
    segment_id: int
    facet_id: int
    vertex_ids: tuple[int, int, int]
    status: str
    distance: float
    gap: float
    axis_parameter: float
    barycentric: tuple[float, float, float]
    axis_point_relative_m: tuple[float, float, float]
    triangle_point_relative_m: tuple[float, float, float]
    normal: tuple[float, float, float] | None
    selected_candidate: int
    tied_candidates: tuple[int, ...]
    axis_span: tuple[float, float] | None


@dataclass(frozen=True)
class TriangleDistancePacket:
    schema: str
    owner: str
    generation: int
    mesh_key: str
    revision: int
    snapshot_id: str
    input_binding: str
    inputs: MappingProxyType
    uploaded: MappingProxyType
    conversion_deltas: MappingProxyType
    pair_origins_relative_m: np.ndarray
    raw: MappingProxyType
    rows: tuple[TriangleDistanceRow, ...]


class DeviceTriangleDistanceBatch:
    """Own an explicit current-pose distance query; no automatic HOST fallback."""

    def __init__(
        self,
        *,
        mode,
        vertex_capacity,
        face_capacity,
        segment_capacity,
        pair_capacity,
        cache_root=None,
        runtime=None,
    ):
        if mode not in ("interpret", "cuda"):
            raise ValueError("mode must be interpret or cuda")
        capacities = dict(
            vertex_capacity=vertex_capacity,
            face_capacity=face_capacity,
            segment_capacity=segment_capacity,
            pair_capacity=pair_capacity,
        )
        for key, value in capacities.items():
            if type(value) is not int or not 1 <= value <= _CAPS[key]:
                raise ValueError(f"{key} capacity must be in [1,{_CAPS[key]}]")
        import tyx as tx

        self._tx = tx
        self._mode = mode
        self._capacities = capacities
        self._owned_runtime = runtime is None
        self._runtime = (
            runtime
            if runtime is not None
            else tx.Runtime(backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root)
        )
        self._runtime_resources = DeviceResources()
        self._runtime_resources.set_runtime(
            self._runtime if self._owned_runtime else _RuntimeLease(self._runtime)
        )
        self._query_resources = None
        self._owner = uuid.uuid4().hex
        self._generation = 0
        self._lock = RLock()
        self._closed = False
        self._last_packet = None
        self._last_attempt_raw = None
        self._counts = dict(
            allocations=0,
            live_bytes=0,
            peak_owned_bytes=0,
            uploads=0,
            uploaded_bytes=0,
            downloads=0,
            downloaded_bytes=0,
            synchronizations=0,
            launches=0,
            completed_launches=0,
            attempts=0,
            queries=0,
            failed_queries=0,
        )

    @property
    def stats(self):
        counts = dict(self._counts)
        counts["attempt_array_bytes"] = sum(
            a.nbytes for a in (self._last_attempt_raw or {}).values()
        )
        packet = self._last_packet
        counts["packet_array_bytes"] = (
            0
            if packet is None
            else (
                packet.pair_origins_relative_m.nbytes
                + sum(
                    a.nbytes
                    for m in (packet.inputs, packet.uploaded, packet.conversion_deltas, packet.raw)
                    for a in m.values()
                )
            )
        )
        return MappingProxyType(counts)

    @property
    def last_packet(self):
        return self._last_packet

    @property
    def last_attempt_raw(self):
        return self._last_attempt_raw

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _check_release(self):
        if self._query_resources is not None and not self._query_resources.closed:
            # UNKNOWN_RELEASE reports the original error without retrying free.
            self._query_resources.close()
        if self._runtime_resources.state == "UNKNOWN_RELEASE":
            self._runtime_resources.close()

    def _new_query_resources(self):
        self._check_release()
        resources = DeviceResources()
        resources.set_runtime(_RuntimeLease(self._runtime))
        self._query_resources = resources
        return resources

    def close(self):
        with self._lock:
            self._check_release()
            if not self._closed:
                self._runtime_resources.close()
                self._closed = True

    def _prepare(
        self,
        *,
        mesh_key,
        revision,
        vertex_ids,
        facet_ids,
        faces,
        vertices,
        segment_ids,
        segments,
        radii,
        candidates,
        origin_m,
        snapshot_id,
    ):
        if (
            not isinstance(mesh_key, str)
            or not mesh_key
            or not isinstance(snapshot_id, str)
            or not snapshot_id
        ):
            raise ValueError("mesh_key and snapshot_id must be nonempty strings")
        if type(revision) is not int or revision < 0:
            raise ValueError("revision must be nonnegative int")
        # Bound all input array dimensions before making immutable byte copies.
        for key, value in (
            ("vertex_capacity", vertex_ids),
            ("face_capacity", facet_ids),
            ("segment_capacity", segment_ids),
            ("pair_capacity", candidates),
        ):
            if not isinstance(value, np.ndarray) or value.ndim < 1:
                raise ValueError("invalid capacity-bearing array")
            if len(value) > self._capacities[key]:
                raise ValueError(f"{key} capacity exceeded")
        arrays = dict(
            vertex_ids=_ids(vertex_ids, "vertex_ids"),
            facet_ids=_ids(facet_ids, "facet_ids"),
            segment_ids=_ids(segment_ids, "segment_ids"),
        )
        nv, nf, ns = len(arrays["vertex_ids"]), len(arrays["facet_ids"]), len(arrays["segment_ids"])
        if nv < 3 or nf < 1 or ns < 1:
            raise ValueError("nonempty vertex/face/segment inputs required")
        arrays.update(
            faces=_array(faces, np.int64, (nf, 3), "faces"),
            vertices=_array(vertices, np.float64, (nv, 3), "vertices"),
            segments=_array(segments, np.float64, (ns, 2, 3), "segments"),
            radii=_array(radii, np.float64, (ns,), "radii"),
            candidates=_array(candidates, np.int64, (None, 2), "candidates"),
            origin_m=_array(origin_m, np.float64, (3,), "origin_m"),
        )
        npairs = len(arrays["candidates"])
        for key, count in zip(_CAPS, (nv, nf, ns, npairs), strict=True):
            if count > self._capacities[key]:
                raise ValueError(f"{key} capacity exceeded")
        if np.any(arrays["faces"] < 0) or np.any(arrays["faces"] >= nv):
            raise ValueError("face vertex row out of range")
        if np.any(arrays["radii"] < 0):
            raise ValueError("negative radius")
        if len(set(map(tuple, arrays["candidates"]))) != npairs:
            raise ValueError("duplicate candidate")
        segment_map = {int(value): i for i, value in enumerate(arrays["segment_ids"])}
        facet_map = {int(value): i for i, value in enumerate(arrays["facet_ids"])}
        geometry = np.empty((npairs, 5, 3), np.float64)
        frames = np.empty((npairs, 3), np.float64)
        scales = np.empty(npairs, np.float64)
        pair_radii = np.empty(npairs, np.float64)
        owners = np.empty((npairs, 2), np.int32)
        for ordinal, (sid, fid) in enumerate(arrays["candidates"]):
            if int(sid) not in segment_map or int(fid) not in facet_map:
                raise ValueError("unknown candidate owner")
            si, fi = segment_map[int(sid)], facet_map[int(fid)]
            tri = arrays["vertices"][arrays["faces"][fi]]
            axis = arrays["segments"][si]
            anchor = tri[0]
            geometry[ordinal, :3] = tri - anchor
            geometry[ordinal, 3:] = axis - anchor
            frames[ordinal] = anchor - arrays["origin_m"]
            edges = np.roll(tri, -1, axis=0) - tri
            length = max(
                *(float(np.linalg.norm(e)) for e in edges),
                float(np.linalg.norm(axis[1] - axis[0])),
                float(arrays["radii"][si]),
            )
            if not np.isfinite(length) or not 2**-30 <= length <= 2**30:
                raise ValueError("local scale outside bounded arithmetic admission")
            scales[ordinal] = length
            pair_radii[ordinal] = arrays["radii"][si]
            owners[ordinal] = (si, fi)
        with np.errstate(over="ignore", invalid="ignore"):
            uploaded = dict(
                geometry=geometry.astype(np.float32),
                radii=pair_radii.astype(np.float32),
                scales=scales.astype(np.float32),
                owners=owners,
            )
        for name, value in uploaded.items():
            if not np.all(np.isfinite(value)):
                raise ValueError(f"source precision overflow in {name}")
        if not np.all(np.isfinite(frames)):
            raise ValueError("source frame overflow")
        deltas = dict(
            geometry=uploaded["geometry"].astype(np.float64) - geometry,
            radii=uploaded["radii"].astype(np.float64) - pair_radii,
            scales=uploaded["scales"].astype(np.float64) - scales,
        )
        for i, length in enumerate(scales):
            if (
                np.max(np.abs(deltas["geometry"][i])) > 8 * _U * length
                or abs(deltas["radii"][i]) > 8 * _U * length
            ):
                raise ValueError("source precision exceeds pair-local admission")
        metadata = dict(mesh_key=mesh_key, revision=revision, snapshot_id=snapshot_id)
        return metadata, arrays, _binding(metadata, arrays), uploaded, deltas, frames, scales

    def _allocate(self, resources, value=None, *, shape=None, dtype=None):
        if value is not None:
            shape, dtype = value.shape, value.dtype
        size = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if self._mode == "cuda":
            array = (
                self._tx.cuda_array(value)
                if value is not None
                else self._tx.cuda_empty(shape, dtype)
            )
        else:
            array = value.copy() if value is not None else np.empty(shape, dtype)
        self._counts["allocations"] += 1
        self._counts["live_bytes"] += size
        self._counts["peak_owned_bytes"] = max(
            self._counts["peak_owned_bytes"], self._counts["live_bytes"]
        )
        resources.add_buffer(_BufferLease(array, self._mode == "cuda", size, self._counts))

        if self._mode == "cuda" and value is not None:
            self._counts["uploads"] += 1
            self._counts["uploaded_bytes"] += size
        return array

    def _download(self, array, label):
        del label
        if self._mode == "cuda":
            result = array.to_numpy()
            self._counts["downloads"] += 1
            self._counts["downloaded_bytes"] += result.nbytes
            return result
        return array.copy()

    def _build(self, prepared, raw, generation):
        metadata, inputs, binding, uploaded, deltas, frames, scales = prepared
        rows, deltas = validate_triangle_distance_rows(
            inputs, uploaded, raw, frames, scales, generation, deltas
        )
        return TriangleDistancePacket(
            "phymeo-device-triangle-distance-v1",
            self._owner,
            generation,
            metadata["mesh_key"],
            metadata["revision"],
            metadata["snapshot_id"],
            binding,
            MappingProxyType(inputs),
            _mapping(uploaded),
            _mapping(deltas),
            _immutable(frames),
            _mapping(raw),
            tuple(rows),
        )

    def query(self, **inputs):
        with self._lock:
            self._check_release()
            if self._closed:
                raise RuntimeError("batch is closed")
            prepared = self._prepare(**inputs)
            count = len(prepared[1]["candidates"])
            if self._generation >= np.iinfo(np.int32).max:
                raise RuntimeError("generation exhausted")
            generation = self._generation + 1
            self._counts["attempts"] += 1
            self._last_attempt_raw = None
            raw = {}
            resources = self._new_query_resources()
            try:
                try:
                    with self._tx.use_runtime(self._runtime):
                        if count:
                            from ._tyx_triangle_distance import triangle_distances

                            device = {
                                key: self._allocate(resources, value)
                                for key, value in prepared[3].items()
                            }
                            outputs = {
                                "candidates": self._allocate(
                                    resources, shape=(count, 18, 11), dtype=np.float32
                                ),
                                "valid": self._allocate(
                                    resources, shape=(count, 18), dtype=np.int32
                                ),
                                "summary": self._allocate(
                                    resources, shape=(count, 16), dtype=np.float32
                                ),
                                "meta": self._allocate(resources, shape=(count, 7), dtype=np.int32),
                            }
                            self._counts["launches"] += 1
                            triangle_distances(
                                count,
                                generation,
                                device["geometry"],
                                device["radii"],
                                device["scales"],
                                device["owners"],
                                outputs["candidates"],
                                outputs["valid"],
                                outputs["summary"],
                                outputs["meta"],
                                mode=self._mode,
                            )
                            if self._mode == "cuda":
                                self._runtime.sync()
                                self._counts["synchronizations"] += 1
                            self._counts["completed_launches"] += 1
                            for key, array in outputs.items():
                                raw[key] = self._download(array, key)
                                self._last_attempt_raw = _mapping(raw)
                        else:
                            raw = {
                                "candidates": np.empty((0, 18, 11), np.float32),
                                "valid": np.empty((0, 18), np.int32),
                                "summary": np.empty((0, 16), np.float32),
                                "meta": np.empty((0, 7), np.int32),
                            }
                            self._last_attempt_raw = _mapping(raw)
                        packet = self._build(prepared, raw, generation)
                finally:
                    resources.close()
                # Resource cleanup and every read/validation must succeed first.
                self._last_packet = packet
                self._generation = generation
                self._counts["queries"] += 1
                return packet
            except BaseException:
                self._counts["failed_queries"] += 1
                raise

    def require_current(self, packet, **inputs):
        with self._lock:
            self._check_release()
            if self._closed:
                raise RuntimeError("batch is closed")
            if not isinstance(packet, TriangleDistancePacket) or packet.owner != self._owner:
                raise ValueError("packet owner mismatch")
            if packet is not self._last_packet:
                raise ValueError("packet is not current")
            if self._prepare(**inputs)[2] != packet.input_binding:
                raise ValueError("packet input binding mismatch")


def validate_triangle_distance_rows(inputs, uploaded, raw, frames, scales, generation, deltas):
    """Validate emitted candidate/witness records without generating geometry.

    Shared by explicit HOST-packed COL2 and resident device-packed M1.
    This routine only checks the complete already emitted finite list.
    """
    count = len(inputs["candidates"])
    expected = {
        "candidates": ((count, 18, 11), np.dtype(np.float32)),
        "valid": ((count, 18), np.dtype(np.int32)),
        "summary": ((count, 16), np.dtype(np.float32)),
        "meta": ((count, 7), np.dtype(np.int32)),
    }
    for key, (shape, dtype) in expected.items():
        value = raw[key]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != dtype
            or not np.all(np.isfinite(value))
        ):
            raise RuntimeError(f"invalid device {key} readback")
    rows = []
    deltas = dict(deltas)
    deltas["published_axis"] = np.zeros((count, 3), np.float64)
    deltas["published_triangle"] = np.zeros((count, 3), np.float64)
    for i, (sid, fid) in enumerate(inputs["candidates"]):
        state, selected, mask, normal_defined, tag, si, fi = map(int, raw["meta"][i])
        if tag != generation or (si, fi) != tuple(map(int, uploaded["owners"][i])):
            raise RuntimeError("device row ownership or generation mismatch")
        if (
            state not in (1, 2, 3)
            or normal_defined not in (0, 1)
            or mask < 0
            or mask >= 1 << 18
        ):
            raise RuntimeError("invalid device row status")
        valid = raw["valid"][i]
        if np.any((valid != 0) & (valid != 1)):
            raise RuntimeError("invalid candidate validity")
        values = raw["candidates"][i]
        summary = raw["summary"][i]
        length = float(scales[i])
        g = uploaded["geometry"][i].astype(np.float64)
        for c in range(18):
            if not valid[c]:
                if np.any(values[c] != 0):
                    raise RuntimeError("invalid slot has nonzero candidate data")
                continue
            s, b0, b1, b2, d2 = map(float, values[c, :5])
            if (
                not 0 <= s <= 1
                or min(b0, b1, b2) < 0
                or abs(b0 + b1 + b2 - 1) > 8 * _U
                or d2 < 0
            ):
                raise RuntimeError("invalid finite candidate address")
            ap = g[3] + s * (g[4] - g[3])
            tp = np.array((b0, b1, b2)) @ g[:3]
            if (
                np.linalg.norm(ap - values[c, 5:8]) > 2e-4 * length
                or np.linalg.norm(tp - values[c, 8:11]) > 2e-4 * length
            ):
                raise RuntimeError("candidate material reconstruction mismatch")
            difference = values[c, 8:11].astype(np.float64) - values[c, 5:8]
            if abs(d2 - float(difference @ difference)) > 32 * _U * max(length * length, d2):
                raise RuntimeError("candidate distance reconstruction mismatch")
        if selected == -1:
            if (
                np.any(valid)
                or state == 1
                or mask
                or normal_defined
                or np.any(summary[:15] != 0)
            ):
                raise RuntimeError("missing selected candidate")
            ties = ()
            span = None
        else:
            if not 0 <= selected < 18 or not valid[selected] or not (mask & (1 << selected)):
                raise RuntimeError("invalid selected candidate")
            if not np.array_equal(summary[2:6], values[selected, :4]) or not np.array_equal(
                summary[6:12], values[selected, 5:11]
            ):
                raise RuntimeError("selected witness publication mismatch")
            best = float(values[selected, 4])
            # Check the emitted finite list; do not calculate new candidates.
            if any(float(values[c, 4]) < best for c in range(18) if valid[c]):
                raise RuntimeError("selected candidate is not the emitted minimum")
            ties = tuple(c for c in range(18) if mask & (1 << c))
            tie_limit = float(
                np.float32(
                    np.float32(16 * _U)
                    * np.float32(uploaded["scales"][i] * uploaded["scales"][i])
                )
            )
            for c in range(18):
                claimed = c in ties
                expected_tie = bool(
                    valid[c]
                    and abs(float(np.float32(values[c, 4] - values[selected, 4]))) <= tie_limit
                )
                if claimed != expected_tie:
                    raise RuntimeError("incomplete candidate tie mask")
            span = (
                min(float(values[c, 0]) for c in ties),
                max(float(values[c, 0]) for c in ties),
            )
            distance = float(summary[0])
            if distance < 0 or abs(distance * distance - best) > 32 * _U * max(
                length * length, best
            ):
                raise RuntimeError("selected distance mismatch")
            if abs(float(summary[1]) - (distance - float(uploaded["radii"][i]))) > 8 * _U * max(
                length, distance
            ):
                raise RuntimeError("radius gap mismatch")
        if float(summary[15]) != float(uploaded["scales"][i]):
            raise RuntimeError("device local scale mismatch")
        normal_floor = float(np.float32(np.float32(32 * _U) * uploaded["scales"][i]))
        expected_normal = bool(state == 1 and selected >= 0 and summary[0] > normal_floor)
        if bool(normal_defined) != expected_normal:
            raise RuntimeError("undefined normal mismatch")
        normal = None
        if expected_normal:
            direction = (summary[9:12].astype(np.float64) - summary[6:9]) / float(summary[0])
            if (
                np.linalg.norm(direction - summary[12:15]) > 2e-3
                or abs(np.linalg.norm(summary[12:15]) - 1) > 2e-3
            ):
                raise RuntimeError("device direction mismatch")
            normal = tuple(map(float, summary[12:15]))
        elif np.any(summary[12:15] != 0):
            raise RuntimeError("unexpected normal for unresolved row")
        axis_point = frames[i] + summary[6:9].astype(np.float64)
        triangle_point = frames[i] + summary[9:12].astype(np.float64)
        deltas["published_axis"][i] = (axis_point - frames[i]) - summary[6:9]
        deltas["published_triangle"][i] = (triangle_point - frames[i]) - summary[9:12]
        if (
            np.linalg.norm(deltas["published_axis"][i]) > 2e-4 * length
            or np.linalg.norm(deltas["published_triangle"][i]) > 2e-4 * length
        ):
            raise RuntimeError("pair frame publication loses local witness")
        rows.append(
            TriangleDistanceRow(
                int(sid),
                int(fid),
                tuple(map(int, inputs["vertex_ids"][inputs["faces"][fi]])),
                {1: "RESOLVED", 2: "CONDITIONING", 3: "DEGENERATE"}[state],
                float(summary[0]),
                float(summary[1]),
                float(summary[2]),
                tuple(map(float, summary[3:6])),
                tuple(map(float, axis_point)),
                tuple(map(float, triangle_point)),
                normal,
                selected,
                ties,
                span,
            )
        )
    return tuple(rows), deltas
