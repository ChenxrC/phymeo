"""Explicit tyx broad phase with model-owned, reusable scratch allocations."""

from contextlib import ExitStack

import numpy as np


class TyxCollisionBatch:
    def __init__(self, mode, offsets, tags, pairs, cache_root):
        import tyx as tx

        from ._tyx_collision_kernels import (
            classify_pairs,
            reduce_bounds,
            round_entry_times,
            surface_entry_times,
            triangle_entry_times,
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
        self._sweep_counts = dict(
            triangle_vertex_face_features=0,
            triangle_edge_edge_features=0,
            triangle_completed_launches=0,
            triangle_proposed_hits=0,
            triangle_edge_edge_proposed_hits=0,
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
                if self._pair_count:
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

    def query(self, envelopes):
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
                    proposed = int(np.count_nonzero(active >= 0))
                    counts["sweep_proposed_hits"] += proposed
                    counts["sweep_features"] += len(page)
                    counts["sweep_completed_launches"] += 1
                    if width == 28:
                        counts["surface_features"] += len(page)
                        counts["surface_proposed_hits"] += proposed
                        counts["surface_completed_launches"] += 1
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

    def stats(self):
        return {**self._counts, **self._sweep_counts, "backend": self._mode}

    def close(self):
        self._resources.close()
