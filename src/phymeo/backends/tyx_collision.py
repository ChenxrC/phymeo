"""Explicit tyx broad phase with model-owned, reusable scratch allocations."""

from contextlib import ExitStack

import numpy as np

from ..constraints.collision_shapes import MAX_CONVEX_VERTICES


class TyxCollisionBatch:
    def __init__(self, mode, offsets, tags, pairs, cache_root, *, spatial=None):
        import tyx as tx

        from ._tyx_collision_kernels import (
            classify_pairs,
            hull_projection_intervals,
            reduce_bounds,
            round_entry_times,
            surface_entry_times,
            triangle_entry_times,
        )
        from ._tyx_heightfield_candidates import HeightfieldCandidateBuffer
        from ._tyx_spatial_kernels import sap_candidates

        self._sap_kernel = sap_candidates
        self._spatial = spatial
        self._tree_buffers = {}
        self._spatial_counts = dict(
            spatial_completed_launches=0,
            spatial_uploads=0,
            spatial_uploaded_bytes=0,
            spatial_downloads=0,
            spatial_downloaded_bytes=0,
            spatial_synchronizations=0,
        )
        self._tx = tx
        self._mode = mode
        self._reduce, self._classify = reduce_bounds, classify_pairs
        self._count, self._pair_count = len(tags), len(pairs)
        self._entry_kernels = {
            5: round_entry_times,
            28: surface_entry_times,
            19: triangle_entry_times,
        }
        self._entry_storage = {}
        self._hull_kernel = hull_projection_intervals
        self._hull_storage = None
        self._sweep_counts = dict(
            arc_chart_features=0,
            arc_chart_completed_launches=0,
            arc_chart_proposed_hits=0,
            hull_axis_terms=0,
            hull_nonempty_intervals=0,
            hull_launches=0,
            hull_completed_launches=0,
            hull_uploads=0,
            hull_uploaded_bytes=0,
            hull_downloads=0,
            hull_downloaded_bytes=0,
            hull_synchronizations=0,
            triangle_vertex_face_features=0,
            triangle_edge_edge_features=0,
            triangle_completed_launches=0,
            triangle_proposed_hits=0,
            triangle_edge_edge_proposed_hits=0,
            capsule_shaft_vertex_features=0,
            capsule_shaft_vertex_proposed_hits=0,
            capsule_shaft_edge_features=0,
            capsule_shaft_edge_proposed_hits=0,
            surface_features=0,
            surface_completed_launches=0,
            surface_proposed_hits=0,
            sweep_allocations=0,
            sweep_features=0,
            sweep_launches=0,
            sweep_completed_launches=0,
            sweep_proposed_hits=0,
            sweep_uploads=0,
            sweep_downloads=0,
            sweep_uploaded_bytes=0,
            sweep_downloaded_bytes=0,
            sweep_synchronizations=0,
        )
        self._heightfield_candidates = HeightfieldCandidateBuffer(self, offsets)
        self._resources = ExitStack()
        self._counts = dict(
            allocations=0,
            static_uploads=0,
            dynamic_uploads=0,
            downloads=0,
            uploaded_bytes=0,
            downloaded_bytes=0,
            launches=0,
            completed_launches=0,
            synchronizations=0,
        )
        self._runtime = tx.Runtime(
            backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root
        )
        self._resources.callback(self._runtime.close)
        try:
            with tx.use_runtime(self._runtime):
                if self._count:
                    self._offsets = self._static(offsets)
                    self._envelopes = self._scratch((int(offsets[-1]), 6), np.float32)
                    self._bounds = self._scratch((self._count, 6), np.float32)
                if self._pair_count and spatial is not None:
                    self._allowed = self._static(spatial.allowed)
                    if spatial.mode == "sap":
                        self._spatial_meta = self._static(spatial.meta)
                        self._order = self._scratch((self._count, 2), np.int32)
                        self._packed = self._scratch((self._count, spatial.words + 2), np.int32)
                elif self._pair_count:
                    self._tags = self._static(tags)
                    self._pairs = self._static(pairs)
                    self._flags = self._scratch((self._pair_count,), np.int32)
        except BaseException:
            self.close()
            raise

    def _static(self, array):
        self._counts["allocations"] += 1
        if self._mode != "cuda":
            return array.copy()
        device = self._tx.cuda_array(array)
        self._resources.callback(device.free)
        self._counts["static_uploads"] += 1
        self._counts["uploaded_bytes"] += array.nbytes
        return device

    def _scratch(self, shape, dtype):
        self._counts["allocations"] += 1
        if self._mode != "cuda":
            return np.empty(shape, dtype=dtype)
        device = self._tx.cuda_empty(shape, dtype)
        self._resources.callback(device.free)
        return device

    def _read(self, array):
        if self._mode != "cuda":
            return array.copy()
        host = array.to_numpy()
        self._counts["downloads"] += 1
        self._counts["downloaded_bytes"] += host.nbytes
        return host

    def _spatial_upload(self, device, host):
        if self._mode != "cuda":
            np.copyto(device, host)
            return
        device.copy_from(host)
        self._counts["dynamic_uploads"] += 1
        self._counts["uploaded_bytes"] += host.nbytes
        self._spatial_counts["spatial_uploads"] += 1
        self._spatial_counts["spatial_uploaded_bytes"] += host.nbytes

    def _spatial_output(self, output):
        if self._mode == "cuda":
            self._runtime.sync()
            self._counts["synchronizations"] += 1
            self._spatial_counts["spatial_synchronizations"] += 1
        raw = self._read(output)
        if self._mode == "cuda":
            self._spatial_counts["spatial_downloads"] += 1
            self._spatial_counts["spatial_downloaded_bytes"] += raw.nbytes
        return raw

    def tree_hits(self, key, tree, leaves, queries, params, shape):
        from ._tyx_collision_bvh import TreeBuffers

        with self._tx.use_runtime(self._runtime):
            if key not in self._tree_buffers:
                self._tree_buffers[key] = TreeBuffers(self, tree, self._count, shape)
            return self._tree_buffers[key].hits(tree, leaves, queries, params, shape)

    def heightfield_face_candidates(self, terrain, sphere, rows, columns):
        return self._heightfield_candidates.select(terrain, sphere, rows, columns)

    def _spatial_query(self, trees, pending):
        # Public bounds must be returned on all backends. Reuse that transfer
        # for the deterministic host endpoint order used by the tyx traversal.
        if self._mode == "cuda":
            self._runtime.sync()
            self._counts["synchronizations"] += 1
        bounds = self._read(self._bounds)
        self._counts["completed_launches"] += 1
        if not self._pair_count:
            return bounds, np.empty(0, np.int32)
        space = self._spatial
        if space.mode == "bvh":
            return bounds, space.tree_flags(bounds, trees, pending, self)
        self._spatial_upload(self._order, space.order(bounds))
        self._counts["launches"] += 1
        self._sap_kernel(
            self._count,
            space.words,
            self._spatial_meta,
            self._allowed,
            self._order,
            self._bounds,
            self._packed,
            mode=self._mode,
        )
        raw = self._spatial_output(self._packed)
        self._counts["completed_launches"] += 1
        self._spatial_counts["spatial_completed_launches"] += 1
        return bounds, space.decode_sap(raw)

    def query(self, envelopes, *, trees=None, pending=None):
        if not self._count:
            return np.empty((0, 6), np.float32), np.empty(0, np.int32)
        with self._tx.use_runtime(self._runtime):
            if self._mode == "cuda":
                self._envelopes.copy_from(envelopes)
                self._counts["dynamic_uploads"] += 1
                self._counts["uploaded_bytes"] += envelopes.nbytes
            else:
                np.copyto(self._envelopes, envelopes)
            self._counts["launches"] += 1
            self._reduce(self._count, self._offsets, self._envelopes, self._bounds, mode=self._mode)
            if self._spatial is not None:
                return self._spatial_query(trees, pending)
            if self._pair_count:
                self._counts["launches"] += 1
                self._classify(
                    self._pair_count,
                    self._pairs,
                    self._tags,
                    self._bounds,
                    self._flags,
                    mode=self._mode,
                )
            if self._mode == "cuda":
                self._runtime.sync()
                self._counts["synchronizations"] += 1
            self._counts["completed_launches"] += 1 + int(self._pair_count > 0)
            return self._read(self._bounds), self._read(
                self._flags
            ) if self._pair_count else np.empty(0, np.int32)

    def entry_times(self, features):
        """Batch all active features using fixed-size pages across queries."""
        if not features:
            return []
        groups = {}
        for index, f in enumerate(features):
            packed = f.packed()
            groups.setdefault(len(packed), []).append((index, packed))
        results = [None] * len(features)
        counts = self._sweep_counts
        with self._tx.use_runtime(self._runtime):
            for width, rows in groups.items():
                kernel = self._entry_kernels[width]
                if width not in self._entry_storage:
                    cap = min(512, 9 * self._pair_count)
                    host = np.zeros((cap, width), dtype=np.float32)
                    self._entry_storage[width] = (
                        host,
                        self._scratch(host.shape, np.float32),
                        self._scratch((cap,), np.float32),
                    )
                    counts["sweep_allocations"] += 2
                host, inputs, output = self._entry_storage[width]
                for offset in range(0, len(rows), len(host)):
                    page = rows[offset : offset + len(host)]
                    for i, (_, packed) in enumerate(page):
                        host[i] = packed
                    if self._mode == "cuda":
                        inputs.copy_from(host)
                        counts["sweep_uploads"] += 1
                        counts["sweep_uploaded_bytes"] += host.nbytes
                    else:
                        np.copyto(inputs[: len(page)], host[: len(page)])
                    counts["sweep_launches"] += 1
                    kernel(len(page), inputs, output, mode=self._mode)
                    if self._mode == "cuda":
                        self._runtime.sync()
                        counts["sweep_synchronizations"] += 1
                        raw = output.to_numpy()
                        counts["sweep_downloads"] += 1
                        counts["sweep_downloaded_bytes"] += raw.nbytes
                    else:
                        raw = output
                    active = raw[: len(page)]
                    if not np.isfinite(active).all() or np.any(
                        (active != -1) & ((active < 0) | (active > 1))
                    ):
                        raise RuntimeError("invalid tyx feature entry proposal")
                    for (index, _), seed in zip(page, active, strict=True):
                        results[index] = float(seed)
                        if getattr(features[index], "arc_chart", False):
                            counts["arc_chart_features"] += 1
                            counts["arc_chart_proposed_hits"] += int(seed >= 0)
                        part = getattr(features[index], "capsule_feature", None)
                        if part is not None:
                            counts[f"capsule_shaft_{part}_features"] += 1
                            counts[f"capsule_shaft_{part}_proposed_hits"] += int(seed >= 0)
                    proposed = int(np.count_nonzero(active >= 0))
                    counts["sweep_proposed_hits"] += proposed
                    counts["sweep_features"] += len(page)
                    counts["sweep_completed_launches"] += 1
                    if width == 28:
                        surface = [
                            seed
                            for (index, _), seed in zip(page, active, strict=True)
                            if not getattr(features[index], "arc_chart", False)
                        ]
                        counts["arc_chart_completed_launches"] += int(len(surface) != len(page))
                        counts["surface_features"] += len(surface)
                        counts["surface_proposed_hits"] += sum(int(seed >= 0) for seed in surface)
                        counts["surface_completed_launches"] += int(bool(surface))
                    elif width == 19:
                        edges = [
                            (features[index].edge_pair, seed)
                            for (index, _), seed in zip(page, active, strict=True)
                        ]
                        edge_count = sum(edge for edge, _ in edges)
                        counts["triangle_edge_edge_features"] += edge_count
                        counts["triangle_vertex_face_features"] += len(page) - edge_count
                        counts["triangle_edge_edge_proposed_hits"] += sum(
                            1 for edge, seed in edges if edge and seed >= 0
                        )
                        counts["triangle_completed_launches"] += 1
                        counts["triangle_proposed_hits"] += proposed
        return results

    def hull_intervals(self, points, na, nb, directions):
        """Return one checked-format FP32 interval proposal for each SAT direction."""
        counts = self._sweep_counts
        with self._tx.use_runtime(self._runtime):
            if self._hull_storage is None:
                cloud = np.zeros((2 * MAX_CONVEX_VERTICES + 1, 3), dtype=np.float32)
                axes = np.zeros((256, 3), dtype=np.float32)
                self._hull_storage = (
                    cloud,
                    axes,
                    self._scratch(cloud.shape, np.float32),
                    self._scratch(axes.shape, np.float32),
                    self._scratch((len(axes), 2), np.float32),
                )
            cloud, axes, device_cloud, device_axes, output = self._hull_storage
            cloud[: len(points)] = points
            if self._mode == "cuda":
                device_cloud.copy_from(cloud)
                counts["hull_uploads"] += 1
                counts["hull_uploaded_bytes"] += cloud.nbytes
            else:
                np.copyto(device_cloud[: len(points)], points)
            proposals = []
            for start in range(0, len(directions), len(axes)):
                page = directions[start : start + len(axes)]
                axes[: len(page)] = page
                if self._mode == "cuda":
                    device_axes.copy_from(axes)
                    counts["hull_uploads"] += 1
                    counts["hull_uploaded_bytes"] += axes.nbytes
                else:
                    np.copyto(device_axes[: len(page)], page)
                counts["hull_launches"] += 1
                self._hull_kernel(
                    len(page), na, nb, device_cloud, device_axes, output, mode=self._mode
                )
                if self._mode == "cuda":
                    self._runtime.sync()
                    counts["hull_synchronizations"] += 1
                    raw = output.to_numpy()
                    counts["hull_downloads"] += 1
                    counts["hull_downloaded_bytes"] += raw.nbytes
                else:
                    raw = output
                active = raw[: len(page)]
                valid = ((active[:, 0] == -1) & (active[:, 1] == -1)) | (
                    (active[:, 0] >= 0) & (active[:, 1] <= 1) & (active[:, 0] <= active[:, 1])
                )
                if not np.isfinite(active).all() or not valid.all():
                    raise RuntimeError("invalid tyx hull interval proposal")
                proposals.extend(None if lo < 0 else (float(lo), float(hi)) for lo, hi in active)
                counts["hull_completed_launches"] += 1
                counts["hull_axis_terms"] += len(page)
                counts["hull_nonempty_intervals"] += int(np.count_nonzero(active[:, 0] >= 0))
        return proposals

    def stats(self):
        return {
            **self._counts,
            **self._sweep_counts,
            **self._spatial_counts,
            **self._heightfield_candidates.counts,
            "backend": self._mode,
        }

    def close(self):
        self._resources.close()
