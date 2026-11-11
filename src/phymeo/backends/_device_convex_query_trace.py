"""Owned opt-in GJK trace transport. No geometry solver or HOST oracle lives here."""
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np

SCHEMA = "phymeo-convex-gjk-trace-v1"
FLOAT_WIDTH = 416
INT_WIDTH = 96
MAX_PAIRS = 8


def normalize_trace_pairs(pairs):
    from ._device_convex_query import _canonical
    if type(pairs) not in (tuple, list):
        raise TypeError("diagnostic pairs must be an explicit tuple or list")
    if not 1 <= len(pairs) <= MAX_PAIRS:
        raise ValueError("diagnostic selection requires 1..8 pairs")
    result = tuple(_canonical(pair) for pair in pairs)
    if len(set(result)) != len(result):
        raise ValueError("duplicate diagnostic pair")
    return result


def _policy(plan):
    return dict(
        gjk_iterations=plan._gjk, epa_iterations=plan._epa,
        epa_vertex_capacity=plan._vertices, epa_face_capacity=plan._faces,
        selected_pair_limit=MAX_PAIRS, float_width=FLOAT_WIDTH, int_width=INT_WIDTH,
        memory_limit_bytes=plan._memory_limit, pair_capacity=plan._capacity,
        schema=SCHEMA, fw_dot_role="diagnostic-only-not-a-gate",
    )


@dataclass(frozen=True)
class ConvexQueryTrace:
    schema: str
    owner: object
    query: object
    pairs: tuple
    pair_rows: np.ndarray
    shape_rows: np.ndarray
    floats: np.ndarray
    integers: np.ndarray
    policy: Mapping
    stats: Mapping

    def __post_init__(self):
        from ._device_convex_query import _freeze, _owned
        object.__setattr__(self, "pairs", tuple(tuple(pair) for pair in self.pairs))
        # Preserve input dtypes: validation rejects forged wrong types, not casts.
        for name in ("pair_rows", "shape_rows", "floats", "integers"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        for name in ("policy", "stats"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


def validate_trace(plan, trace):
    if not isinstance(trace, ConvexQueryTrace):
        raise TypeError("expected a ConvexQueryTrace")
    if trace.owner is not plan._trace_owner:
        raise ValueError("diagnostic belongs to another query owner")
    if trace.schema != SCHEMA or dict(trace.policy) != _policy(plan):
        raise ValueError("diagnostic policy binding differs")
    canonical = normalize_trace_pairs(trace.pairs)
    if canonical != trace.pairs:
        raise ValueError("diagnostic pair identity is not canonical")
    plan.require_current(trace.query)
    d, h = len(canonical), plan._gjk
    layouts = (
        (trace.pair_rows, np.int32, (d,)), (trace.shape_rows, np.int32, (d, 2)),
        (trace.floats, np.float32, (d, h, FLOAT_WIDTH)),
        (trace.integers, np.int32, (d, h, INT_WIDTH)),
    )
    for array, dtype, shape in layouts:
        if array.dtype != dtype or array.shape != shape or array.flags.writeable:
            raise ValueError("diagnostic wire shape/dtype/ownership differs")
    row_map = {row.pair: i for i, row in enumerate(trace.query.rows)}
    vertex_ids = trace.query.source["vertex_ids"]
    offsets = trace.query.source["vertex_offsets"]
    for j, pair in enumerate(canonical):
        if pair not in row_map or int(trace.pair_rows[j]) != row_map[pair]:
            raise ValueError("diagnostic membership differs from the actual query")
        expected = tuple(trace.query.shape_keys.index(key) for key in pair)
        if tuple(trace.shape_rows[j]) != expected:
            raise ValueError("diagnostic shape/source binding differs")
        row = row_map[pair]
        n = int(trace.query.raw["meta"][row, 4])
        if not 0 <= n <= h:
            raise ValueError("diagnostic terminal GJK count is invalid")
        iv, fv = trace.integers[j], trace.floats[j]
        if np.any(iv[n:] != 0) or np.any(fv[n:] != 0):
            raise ValueError("diagnostic unwritten tail is not empty")
        for iteration in range(n):
            item = iv[iteration]
            if tuple(item[:4]) != (1, trace.query.attempt_tag, row, iteration):
                raise ValueError("diagnostic attempt/iteration binding differs")
            if (
                not 0 <= item[4] <= 4 or not 0 <= item[5] <= 4
                or item[6] not in (-1, 0, 1) or not 0 <= item[7] <= 4
                or not 0 <= item[8] <= 15 or not 1 <= item[9] <= 8
                or item[14] != iteration + 1
                or item[20] not in (0, 1) or not 0 <= item[21] <= 4
                or item[22] not in (0, 1) or item[23] != 0
                or np.any(item[93:] != 0)
            ):
                raise ValueError("invalid diagnostic iteration protocol")
            full_surface = int(any(plan._tags[shape, 0] >= 4 for shape in expected))
            if item[22] != full_surface or fv[iteration, 30] != trace.query.raw["gap"][row, 3]:
                raise ValueError("diagnostic scale/support frame differs")
            if full_surface:
                origin = trace.query.source["pair_origins_relative_m"][row]
                if not np.array_equal(fv[iteration, 412:415], origin):
                    raise ValueError("diagnostic pair anchor differs from accepted source")
            elif iteration and not np.array_equal(
                fv[iteration, 412:415], fv[0, 412:415]
            ):
                raise ValueError("diagnostic kernel anchor changes within a query")
            for side, shape in enumerate(expected):
                slot = int(item[16 + side])
                if not offsets[shape] <= slot < offsets[shape + 1]:
                    raise ValueError("diagnostic support slot is outside source shape")
                if item[18 + side] != vertex_ids[slot]:
                    raise ValueError("diagnostic support material ID differs")
                for count_index, slot_start in ((4, 24), (5, 36)):
                    for k in range(int(item[count_index])):
                        slot = int(item[slot_start + 2*k + side])
                        if not offsets[shape] <= slot < offsets[shape+1]:
                            raise ValueError("diagnostic simplex material slot differs")
            subsets = item[48:93].reshape(15, 3)
            if (
                np.any(subsets[:, 0] < 0) or np.any(subsets[:, 0] > 4)
                or np.any((subsets[:, 1] != 0) & (subsets[:, 1] != 1))
                or np.any(subsets[:, 2] < 0) or np.any(subsets[:, 2] > 3)
            ):
                raise ValueError("invalid diagnostic subset flags")
        # Nonfinite intermediate floats are valid evidence of an arithmetic failure.
        # They are retained as original bytes, never used to validate geometry.
    expected_bytes = d*h*(FLOAT_WIDTH+INT_WIDTH)*4 + d*2*4
    if (
        trace.stats["owned_bytes"] != expected_bytes
        or trace.stats["readback_bytes"] != expected_bytes
        or trace.stats["host_read_bytes"] != expected_bytes
        or trace.stats["clear_launches"] != 1
        or trace.stats["released"] is not True
    ):
        raise ValueError("diagnostic ownership/transfer accounting differs")
    return trace


class _TraceCapture:
    """A temporary transaction, owned under the existing batch lock."""
    def __init__(self, plan, pairs):
        self.plan = plan
        self.pairs = normalize_trace_pairs(pairs)
        self.resources = ExitStack()
        self.buffers = {}
        self.raw = None
        self.result = None
        self.pair_rows = None
        self.shape_rows = None
        self.released = False
        self.read_stats = None
        self.allocation_stats = None
        self.clear_launches = 0
        self.owned_bytes = len(self.pairs)*plan._gjk*(FLOAT_WIDTH+INT_WIDTH)*4 + len(self.pairs)*8

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def prepare(self, actual_pairs):
        plan = self.plan
        rows = {pair: i for i, pair in enumerate(actual_pairs)}
        if any(pair not in rows for pair in self.pairs):
            raise ValueError("diagnostic selection is not present in this query")
        if plan._counts["live_bytes"] + self.owned_bytes > plan._memory_limit:
            raise MemoryError("convex query diagnostic exceeds owned memory limit")
        self.pair_rows = np.asarray([rows[pair] for pair in self.pairs], dtype=np.int32)
        self.shape_rows = np.asarray([
            [plan._shape_index[key] for key in pair] for pair in self.pairs], dtype=np.int32)
        d, h = len(self.pairs), plan._gjk
        before = dict(plan._counts)
        self.buffers["pairs"] = plan._allocate(
            (d, 2), np.int32, self.shape_rows, resources=self.resources)
        self.buffers["floats"] = plan._allocate(
            (d, h, FLOAT_WIDTH), np.float32, resources=self.resources)
        self.buffers["integers"] = plan._allocate(
            (d, h, INT_WIDTH), np.int32, resources=self.resources)
        plan._launch(plan._kernels["trace_clear"], "query_launches", d*h, h,
                     self.buffers["floats"], self.buffers["integers"])
        self.clear_launches += 1
        if sum(value.nbytes for value in self.buffers.values()) != self.owned_bytes:
            raise RuntimeError("diagnostic allocation byte layout differs")
        self.allocation_stats = {
            name: plan._counts[name]-before[name]
            for name in ("allocations", "host_upload_bytes", "uploaded_bytes", "static_uploads", "launches")
        }

    def arguments(self):
        return (len(self.pairs), self.buffers["pairs"],
                self.buffers["floats"], self.buffers["integers"])

    def readback(self):
        plan = self.plan
        before = dict(plan._counts)
        values = dict(
            pairs=plan._read(self.buffers["pairs"]),
            floats=plan._read(self.buffers["floats"]),
            integers=plan._read(self.buffers["integers"]),
        )
        d, h = len(self.pairs), plan._gjk
        for name, dtype, shape in (
            ("pairs", np.int32, (d, 2)),
            ("floats", np.float32, (d, h, FLOAT_WIDTH)),
            ("integers", np.int32, (d, h, INT_WIDTH)),
        ):
            if values[name].dtype != dtype or values[name].shape != shape:
                raise RuntimeError("diagnostic readback wire differs: " + name)
        if not np.array_equal(values["pairs"], self.shape_rows):
            raise RuntimeError("diagnostic uploaded pair rows differ")
        self.raw = values
        self.read_stats = {name: plan._counts[name]-before[name]
                           for name in ("downloads", "downloaded_bytes", "host_read_bytes")}

    def release(self):
        if not self.released:
            # Mark before closing so a failed free is not repeated. ExitStack still
            # visits all registered allocations; publication remains forbidden.
            self.released = True
            self.resources.close()
            self.buffers.clear()

    def build(self, query):
        if not self.released or self.raw is None:
            raise RuntimeError("diagnostic must read and release before publication")
        d, h = len(self.pairs), self.plan._gjk
        result = ConvexQueryTrace(
            SCHEMA, self.plan._trace_owner, query, self.pairs, self.pair_rows,
            self.raw["pairs"], self.raw["floats"], self.raw["integers"], _policy(self.plan),
            dict(self.allocation_stats, **self.read_stats, owned_bytes=self.owned_bytes,
                 readback_bytes=sum(value.nbytes for value in self.raw.values()),
                 clear_launches=self.clear_launches, released=True,
                 host_payload_bytes=sum(value.nbytes for value in self.raw.values())
                                    + self.pair_rows.nbytes,
                 temporary_layout=(("pairs", (d,2), np.dtype(np.int32).str),
                                   ("floats", (d,h,FLOAT_WIDTH), np.dtype(np.float32).str),
                                   ("integers", (d,h,INT_WIDTH), np.dtype(np.int32).str))),
        )
        self.result = validate_trace(self.plan, result)
        return self.result
