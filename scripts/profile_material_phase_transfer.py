"""Measure original same-phase material pages without changing their physics.

Only the normal/friction row law runs on TYX. Geometry, material addressing,
mechanics and public records remain HOST work. API clocks are not GPU events.
"""

import argparse
import gzip
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, is_dataclass
from functools import wraps
from hashlib import sha256
from pathlib import Path
from statistics import median
from threading import get_ident
from time import perf_counter_ns
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PAGE_ROWS = 128
COUNTERS = ("completed_launches", "sites", "uploads", "downloads", "uploaded_bytes", "downloaded_bytes")
API_PARTS = ("upload", "kernel_submit", "sync", "download")
PHASES = ("entry", "midpoint", "endpoint")
MAX_SUMMARY_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class ProfileConfig:
    worlds: int
    faces: int
    mode: str
    case: str = "uniform"
    warm_windows: int = 2
    formal_windows: int = 3
    steps: int = 4
    max_seconds: float = 180.0
    max_raw_bytes: int = 32 * 1024 * 1024

    def __post_init__(self):
        if type(self.worlds) is not int or type(self.faces) is not int:
            raise ValueError("world and face counts must be integers")
        if self.case == "uniform":
            valid = self.worlds in (1, 32) and self.faces in (2, 8)
        else:
            valid = self.case == "heterogeneous" and (self.worlds, self.faces) == (3, 8)
        if not valid or self.mode not in ("cuda", "interpret"):
            raise ValueError("unsupported frozen case or mode")
        for name, low, high in (("warm_windows", 0, 2), ("formal_windows", 1, 3),
                                ("steps", 1, 4), ("max_raw_bytes", 1, 32 * 1024 * 1024)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} outside frozen resource bounds")
        if (isinstance(self.max_seconds, bool) or not isinstance(self.max_seconds, (int, float))
                or not math.isfinite(self.max_seconds) or not 0 < self.max_seconds <= 180):
            raise ValueError("max_seconds must be finite and in (0, 180]")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def summarize_windows(config, *, cold, warmups, formal, complete_output_match, real_cuda):
    """The gate is an upper bound on candidate cost, never a promised speedup."""
    if len(warmups) != config.warm_windows or len(formal) != config.formal_windows:
        raise ValueError("window count differs from the frozen protocol")
    for window in (cold, *warmups, *formal):
        whole = _integer(window["whole_ns"], "whole wall", 1)
        extra = _integer(window["extra_page_api_ns"], "extra page cost")
        total = _integer(window["all_page_api_ns"], "all page cost")
        exclusive = window["exclusive_ns"]
        if (not 0 <= extra <= total <= whole
                or any(type(v) is not int or v < 0 for v in exclusive.values())
                or sum(exclusive.values()) != whole):
            raise ValueError("inconsistent or overlapping cost wall totals")
        for count in window["phase_sites"]:
            _integer(count, "phase site count")
    whole = sum(w["whole_ns"] for w in formal)
    extra = sum(w["extra_page_api_ns"] for w in formal)
    total = sum(w["all_page_api_ns"] for w in formal)
    dense = all(any(s > PAGE_ROWS for s in w["phase_sites"]) for w in formal)
    complete = all(
        w["counter_audit"] == "passed"
        and w["accepted_world_steps"] == config.worlds * config.steps
        and len(w["phase_sites"]) == 3 * config.steps
        for w in formal
    )
    eligible = bool(
        config.mode == "cuda" and real_cuda is True and complete_output_match is True
        and config.steps == 4 and config.warm_windows == 2 and config.formal_windows == 3
        and dense and complete
    )
    ordered = sorted(w["whole_ns"] for w in formal)
    return dict(
        formal_whole_ns=whole,
        formal_extra_page_api_ns=extra,
        formal_all_page_api_ns=total,
        extra_page_fraction_of_whole=extra / whole,
        all_page_fraction_of_whole=total / whole,
        eligible_formal_cuda_dense=eligible,
        measured_cost_gate=eligible and extra * 100 >= whole * 15,
        candidate_cost_scope="extra-page API wall upper bound",
        default_changed=False,
        formal_windows=len(formal),
        p50_ms=median(ordered) / 1e6,
        p95_ms=ordered[math.ceil(.95 * len(ordered)) - 1] / 1e6,
        world_substeps_per_s=config.worlds * config.steps * len(formal) * 1e9 / whole,
    )


def audit_page_counters(sites_by_phase, counters, *, mode):
    if mode not in ("cuda", "interpret"):
        raise ValueError("invalid page counter mode")
    sites = [_integer(s, "phase sites") for s in sites_by_phase]
    pages = [(s + PAGE_ROWS - 1) // PAGE_ROWS for s in sites]
    count = sum(pages)
    cuda = mode == "cuda"
    expected = dict(
        completed_launches=count, sites=sum(sites),
        uploads=count if cuda else 0, downloads=count if cuda else 0,
        uploaded_bytes=6656 * count if cuda else 0,
        downloaded_bytes=2560 * count if cuda else 0,
        explicit_synchronizations=count if cuda else 0,
    )
    for name, value in expected.items():
        if counters.get(name) != value:
            raise ValueError(f"page counter {name}: actual {counters.get(name)} != {value}")
    return dict(status="passed", pages=count, extra_pages=sum(max(p - 1, 0) for p in pages),
                sites=sum(sites), expected=expected)


def partition_wall(begin_ns, end_ns, events, *, caller_thread):
    """Project active stacks onto one wall clock; parked spans never sum into it."""
    if type(begin_ns) is not int or type(end_ns) is not int or begin_ns >= end_ns:
        raise ValueError("invalid wall interval")
    edges = [(begin_ns, 0, 0, "", 0), (end_ns, 0, 0, "", 0)]
    calls, spans, detail = Counter(), [], {}
    for token, (a, b, tid, label) in enumerate(events, 1):
        if type(a) is not int or type(b) is not int or b < a:
            raise ValueError("invalid recorded clock interval")
        a, b = max(begin_ns, a), min(end_ns, b)
        if a >= b:
            continue
        calls[label] += 1
        spans.append((a - begin_ns, b - begin_ns, tid, label))
        detail[token] = (a, b, label)
        edges.extend(((a, 1, tid, label, token), (b, -1, tid, label, token)))
    active, totals, blocked = defaultdict(dict), Counter(), Counter()
    previous = begin_ns

    def leaf(stack, default):
        if not stack:
            return default
        chosen = max(stack, key=lambda t: (detail[t][0], -detail[t][1], t))
        return stack[chosen]

    for at, sign, thread, label, token in sorted(edges):
        if previous < at:
            dt = at - previous
            work = []
            for tid, stack in active.items():
                if tid == caller_thread or not stack:
                    continue
                if "parked" in stack.values():
                    blocked["parked_world_stack_sum"] += dt
                else:
                    work.append(leaf(stack, "world_stack"))
            lead = leaf(active[caller_thread], "scheduler_or_other_caller")
            if lead == "await_world":
                blocked["caller_await_inclusive"] += dt
            if len(work) > 1 or (work and lead not in (
                    "await_world", "schedule_world", "scheduler_or_other_caller")):
                owner = "handoff_overlap"
            elif work:
                owner = work[0]
            else:
                owner = "handoff_idle" if lead == "await_world" else lead
            totals[owner] += dt
        if sign == 1:
            active[thread][token] = label
        elif sign == -1:
            del active[thread][token]
        previous = at
    if sum(totals.values()) != end_ns - begin_ns or any(active.values()):
        raise ValueError("wall clock partition did not close")
    return dict(exclusive_ns=dict(totals), inclusive_wait_ns=dict(blocked),
                calls=dict(calls), main_thread=caller_thread, spans=spans)


def _native(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _native(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: _native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(v) for v in value]
    if type(value).__module__.startswith("numpy") and hasattr(value, "tolist"):
        return value.tolist()
    return value


def canonical_frames(frames):
    """Only SceneSubstepReport.timings values are normalized, with their keys kept."""
    result = deepcopy(_native(frames))
    for frame in result:
        for world in frame["worlds"]:
            for substep in world["result"]["report"]["substeps"]:
                substep["timings"] = {key: 0.0 for key in substep["timings"]}
    return result


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def _digest(value):
    return sha256(_encoded(value)).hexdigest()


def _cases():
    path = str(ROOT / "tests" / "scale")
    if path not in sys.path:
        sys.path.insert(0, path)
    import material_phase_transfer_cases
    return material_phase_transfer_cases


def _delta(after, before):
    return {k: after[k] - before.get(k, 0) for k in after}


def _counts(batch):
    current = batch.stats()
    return {k: int(current.get(k, 0)) for k in COUNTERS}


def _native_counts():
    import tyx
    return asdict(tyx.runtime_stats())


def _audit_native(native, counters, mode, *, cold):
    cuda = mode == "cuda"
    expected = dict(cuda_h2d_bytes=counters["uploaded_bytes"],
                    cuda_d2h_bytes=counters["downloaded_bytes"],
                    cuda_kernel_launches=counters["completed_launches"] if cuda else 0,
                    cuda_d2d_bytes=0)
    extras = {name: native[name] - value for name, value in expected.items()}
    if any(v < 0 for v in extras.values()) or (not cold and any(extras.values())):
        raise ValueError(f"native/layer counter mismatch: {extras}")
    return extras


class _Probe:
    """Temporary call observers; never add device work or alter return values."""

    def __init__(self, batch=None):
        self.batch = batch
        self.main = get_ident()
        self.events, self.phases = [], []
        self.current = None
        self.step = 0
        self.hooks = ExitStack()
        self.layers, self.array_types = set(), set()
        self.device_arrays = {}
        self.device = None
        self.syncs = 0
        self.attempts = Counter()
        self.completed = Counter()
        self.bytes = Counter()

    def timed(self, owner, name, label):
        original = getattr(owner, name)

        @wraps(original)
        def observed(*args, **kwargs):
            start, tid = perf_counter_ns(), get_ident()
            try:
                return original(*args, **kwargs)
            finally:
                self.events.append((start, perf_counter_ns(), tid, label))

        self.hooks.enter_context(patch.object(owner, name, observed))

    def _api(self, original, part, *args, **kwargs):
        phase = self.current
        if phase is None:
            return original(*args, **kwargs)
        if part == "upload":
            phase["page"] += 1
            phase["page_rows"].append({"index": phase["page"], "input": args[1].copy()})
        elif part == "kernel_submit" and phase["mode"] == "interpret":
            phase["page"] += 1
            phase["page_rows"].append({"index": phase["page"]})
        page = phase["page"]
        if page < 0:
            raise RuntimeError("page API occurred before a page started")
        row = phase["page_rows"][page]
        if part == "kernel_submit":
            row["count"] = int(args[0])
            if phase["mode"] == "interpret":
                row["input"] = args[1][:row["count"]].copy()
        label = f"page:{phase['ordinal']}:{page}:{part}"
        self.attempts[part] += 1
        tid = get_ident()
        start = perf_counter_ns()
        try:
            answer = original(*args, **kwargs)
        except BaseException as error:
            stop = perf_counter_ns()
            self.events.append((start, stop, tid, label))
            row["error"] = dict(type=type(error).__name__, message=str(error))
            raise
        stop = perf_counter_ns()
        self.events.append((start, stop, tid, label))
        # Observer bookkeeping and copies belong to HOST residual, not API time.
        self.completed[part] += 1
        if part == "kernel_submit" and phase["mode"] == "interpret":
            row["output"] = args[2][:row["count"]].copy()
        if part == "upload":
            self.bytes["upload"] += args[1].nbytes
        elif part == "download":
            self.bytes["download"] += answer.nbytes
            row["output"] = answer
        elif part == "sync":
            self.syncs += 1
        return answer

    def attach_layer(self, layer):
        if id(layer) in self.layers:
            return
        self.layers.add(id(layer))
        self.device = _device_signature(layer, layer.mode)
        self.device_arrays[id(layer._in)] = (layer, "input")
        self.device_arrays[id(layer._out)] = (layer, "output")
        original_evaluate = layer.evaluate

        def evaluate(rows):
            parent = self.current
            if parent is None:
                phase = dict(ordinal=len(self.phases), phase="row-protocol", step=0,
                             substep=0, requests=(), trace=None)
                self.phases.append(phase)
            else:
                phase = parent
            phase.update(mode=layer.mode, page=-1, page_rows=[], rows=rows, answer=None)
            self.current = phase
            start, tid = perf_counter_ns(), get_ident()
            try:
                phase["answer"] = original_evaluate(rows)
                return phase["answer"]
            finally:
                self.events.append((start, perf_counter_ns(), tid, "law_wrapper"))
                self.current = parent

        self.hooks.enter_context(patch.object(layer, "evaluate", evaluate))
        kernel = layer._kernel
        self.hooks.enter_context(patch.object(layer, "_kernel", lambda *a, **kw: self._api(kernel, "kernel_submit", *a, **kw)))
        sync = layer._runtime.sync
        self.hooks.enter_context(patch.object(layer._runtime, "sync", lambda *a, **kw: self._api(sync, "sync", *a, **kw)))
        if layer.mode == "cuda":
            cls = type(layer._in)
            if cls not in self.array_types:
                self.array_types.add(cls)
                upload, download = cls.copy_from, cls.to_numpy

                def copy_from(device, host, *args, **kwargs):
                    if self.device_arrays.get(id(device), (None, None))[1] == "input":
                        return self._api(upload, "upload", device, host, *args, **kwargs)
                    return upload(device, host, *args, **kwargs)

                def to_numpy(device, *args, **kwargs):
                    if self.device_arrays.get(id(device), (None, None))[1] == "output":
                        return self._api(download, "download", device, *args, **kwargs)
                    return download(device, *args, **kwargs)

                self.hooks.enter_context(patch.object(cls, "copy_from", copy_from))
                self.hooks.enter_context(patch.object(cls, "to_numpy", to_numpy))

    def __enter__(self):
        if self.batch is None:
            return self
        from concurrent.futures import ThreadPoolExecutor
        from phymeo.backends import tyx_multiface_contact as group
        from phymeo.runtime import scene_stepper
        from phymeo.runtime.registry import StateRegistry
        from phymeo.solvers.cloth_link_contact import ClothLinkContacts

        try:
            for owner, name, label in (
                (group._WorldCall, "run", "world_stack"),
                (group._WorldCall, "submit", "parked"),
                (group._WorldCall, "next_event", "await_world"),
                (ThreadPoolExecutor, "submit", "schedule_world"),
                (ClothLinkContacts, "evaluate", "material_assembly_and_scatter"),
                (ClothLinkContacts, "_witness", "pose_geometry"),
                (StateRegistry, "snapshot", "state_and_inputs_snapshot"),
                (StateRegistry, "clone_state", "state_clone"),
                (StateRegistry, "validate_candidate", "candidate_validation"),
                (scene_stepper, "own_diagnostics", "diagnostic_publication"),
            ):
                self.timed(owner, name, label)
            original_factory = group.TyxNormalLayer

            def factory(*args, **kwargs):
                start = perf_counter_ns()
                try:
                    layer = original_factory(*args, **kwargs)
                    self.attach_layer(layer)
                    return layer
                finally:
                    self.events.append((start, perf_counter_ns(), get_ident(), "workspace_setup"))

            self.hooks.enter_context(patch.object(group, "TyxNormalLayer", factory))
            if self.batch._layer is not None:
                self.attach_layer(self.batch._layer)
            original = self.batch._evaluate

            def evaluate(requests, index, capacity):
                substep, phase_index = divmod(index, 3)
                phase = dict(ordinal=len(self.phases), phase=PHASES[phase_index],
                             step=self.step, substep=substep,
                             requests=tuple((k, req.rows, req.faces) for k, req in requests.items()),
                             trace=None, rows=None, answer=None, page_rows=[])
                self.phases.append(phase)
                self.current = phase
                start = perf_counter_ns()
                before, native_before, sync_before = _counts(self.batch), _native_counts(), self.syncs
                try:
                    trace, answer = original(requests, index, capacity)
                    phase["trace"], phase["answer"] = trace, answer
                    return trace, answer
                except BaseException as error:
                    phase["error"] = dict(type=type(error).__name__, message=str(error))
                    raise
                finally:
                    phase["counter_delta"] = _delta(_counts(self.batch), before)
                    phase["counter_delta"]["explicit_synchronizations"] = self.syncs - sync_before
                    phase["native_counter_delta"] = _delta(_native_counts(), native_before)
                    self.events.append((start, perf_counter_ns(), get_ident(), "assemble_pages"))
                    self.current = None

            self.hooks.enter_context(patch.object(self.batch, "_evaluate", evaluate))
            return self
        except BaseException:
            self.hooks.close()
            raise

    def __exit__(self, *args):
        self.hooks.close()

    def receipts(self):
        import numpy as np
        receipts = []
        for phase in self.phases:
            trace = phase["trace"]
            rows = phase["rows"]
            if rows is None:
                source = [r for _, r, _ in phase["requests"]]
                rows = np.concatenate(source) if source else np.empty((0, 13))
            sites = len(rows)
            item = dict(step=phase["step"], substep=phase["substep"], phase=phase["phase"],
                        sites=sites, pages=(sites + 127) // 128,
                        inputs=_native(rows), outputs=_native(phase["answer"]),
                        faces=[] if trace is None else _native(trace.faces),
                        world_offsets=[] if trace is None else _native(trace.world_offsets),
                        page_receipts=[],
                        counter_delta=phase.get("counter_delta"),
                        native_counter_delta=phase.get("native_counter_delta"))
            for row in phase["page_rows"]:
                count = row.get("count", min(128, max(0, sites - 128 * row["index"])))
                item["page_receipts"].append(dict(
                    index=row["index"], active_rows=count,
                    input_fp32=_native(row["input"][:count]) if "input" in row else None,
                    output_fp32=_native(row["output"][:count]) if "output" in row else None,
                    **({"error": row["error"]} if "error" in row else {}),
                ))
            if "error" in phase:
                item["error"] = phase["error"]
            receipts.append(item)
        return receipts



def _device_signature(layer, mode):
    if mode == "interpret":
        return dict(backend=mode, actual_device="HOST", real_cuda=False)
    if layer is None:
        return dict(backend=mode, actual_device=None, real_cuda=False,
                    reason="no nonempty material phase executed")
    from cuda.bindings import driver as cu

    def checked(result):
        if result[0] != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA identity query failed: {result[0]}")
        return result[1]

    ordinal = int(layer._in.device_ordinal)
    if int(layer._out.device_ordinal) != ordinal:
        raise RuntimeError("material buffers span different CUDA devices")
    device = checked(cu.cuDeviceGet(ordinal))
    raw_name = checked(cu.cuDeviceGetName(128, device)).decode("utf8")
    return dict(backend=mode, actual_device=f"cuda:{ordinal}", real_cuda=True,
                device_name=raw_name.split("\0", 1)[0].strip(), device_name_raw=raw_name,
                driver_version=int(checked(cu.cuDriverGetVersion())))


def _check_loaded_sources(root):
    if root is None:
        return {}
    expected = Path(root).resolve() / "src" / "phymeo"
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name == "phymeo" or name.startswith("phymeo."):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(expected):
                    raise RuntimeError(f"Phymeo module outside bound source: {name}: {path}")
                result[name] = str(path)
    if "phymeo" not in result:
        raise RuntimeError("bound Phymeo package was not loaded")
    return result


def _check_phase_rows(receipts, worlds, *, mode):
    import numpy as np
    for receipt in receipts:
        rows = np.asarray(receipt["inputs"], dtype=float).reshape(-1, 13)
        actual = np.asarray(receipt["outputs"], dtype=float).reshape(-1, 5)
        if len(rows) != receipt["sites"] or actual.shape != (len(rows), 5):
            raise ValueError("incomplete phase material rows")
        if receipt["counter_delta"] is not None:
            audit_page_counters([len(rows)], receipt["counter_delta"], mode=mode)
        if len(rows):
            rate = np.sum(rows[:, :3] * rows[:, 3:6], axis=1)
            magnitude = (np.maximum(0, rows[:, 6] - rows[:, 7]) * rows[:, 8]
                         * rows[:, 9] * np.maximum(0, 1 - rows[:, 10] * rate))
            tangent = rows[:, 3:6] - rate[:, None] * rows[:, :3]
            scale = np.maximum(rows[:, 12], np.linalg.norm(tangent, axis=1))
            force = magnitude[:, None] * rows[:, :3] - (
                rows[:, 11] * magnitude / scale)[:, None] * tangent
            expected = np.column_stack((rate, magnitude, force))
            np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=1e-7)
        position = 0
        for key, start, stop in receipt["world_offsets"]:
            if key not in worlds or start != position or not start <= stop <= len(rows):
                raise ValueError("invalid world range in actual material phase")
            for address in (a for a in receipt["faces"] if a["world"] == key):
                world = worlds[key]
                if address["face"] not in world.keys:
                    raise ValueError("unregistered material face")
                face_index = world.keys.index(address["face"])
                if (tuple(address["nodes"]) != world.faces[face_index]
                        or address["robot"] != "link" or address["link"] != "shaft"
                        or address["start"] != position
                        or not position < address["stop"] <= stop):
                    raise ValueError("phase face/node identity changed")
                position = address["stop"]
            if position != stop:
                raise ValueError("phase face ranges do not cover world rows")
        if receipt["world_offsets"] and position != len(rows):
            raise ValueError("phase ranges do not cover all material rows")
        for page in receipt["page_receipts"]:
            start = 128 * page["index"]
            count = page["active_rows"]
            np.testing.assert_array_equal(np.asarray(page["input_fp32"]),
                                          rows[start:start + count].astype(np.float32))
            np.testing.assert_array_equal(np.asarray(page["output_fp32"]),
                                          actual[start:start + count])


def run_row_protocol(count, *, mode, cache_root):
    """Independent fixed-row qualification of the unchanged normal-layer operator."""
    rows, expected = _cases().law_protocol_rows(count)
    if mode not in ("cuda", "interpret"):
        raise ValueError("row protocol mode must be cuda or interpret")
    import numpy as np
    import phymeo as ph

    with ph.TyxNormalLayer(mode, cache_root=cache_root) as layer, _Probe() as probe:
        probe.attach_layer(layer)
        before, native_before = _counts(layer), _native_counts()
        start = perf_counter_ns()
        result = layer.evaluate(rows)
        elapsed = perf_counter_ns() - start
        counts = _delta(_counts(layer), before)
        counts["explicit_synchronizations"] = probe.syncs
        native = _delta(_native_counts(), native_before)
        audit = audit_page_counters([count], counts, mode=mode)
        extra = _audit_native(native, counts, mode, cold=True)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-7)
        receipt = dict(
            scope="arithmetic-row protocol, not a physical trajectory",
            inputs=rows.tolist(), outputs=result.tolist(), expected_outputs=expected.tolist(),
            actual_device=_device_signature(layer, mode), whole_ns=elapsed,
            counter_audit=audit, counter_delta=counts, native_counter_delta=native,
            cold_native_extra=extra, page_receipts=probe.receipts(),
        )
    return receipt


def _input_manifest(worlds, config):
    return dict(
        world_keys=list(worlds), h_s=1 / 256, steps=config.steps,
        declared_layout=_native(_cases().frozen_layout(config.worlds, config.faces, config.case)),
        worlds=[dict(
            world=key, models=_native(world.program.scene.domains),
            program_signature=_native(world.program.problem_signature()),
            reference_position_m=_native(world.reference), areas_m2=_native(world.areas),
            initial=_native(world.initial), controls=_native(world.controls),
            faces=[dict(robot="link", link="shaft", face=name, nodes=list(nodes))
                   for name, nodes in zip(world.keys, world.faces, strict=True)],
        ) for key, world in worlds.items()],
    )


def _physical_checks(worlds, records, config):
    import numpy as np
    from multiface_load_cases import certify

    checks = {}
    layout = _cases().frozen_layout(config.worlds, config.faces, config.case)
    for spec in layout:
        world = worlds[spec.key]
        frames = [record[spec.key] for record in records]
        if not all(frame.report.accepted for frame in frames):
            raise RuntimeError(f"physical world {spec.key} rejected the frozen input")
        if spec.cloth_shift_z_m:
            for frame in frames:
                data = frame.report.substeps[0].diagnostics["robot_contact"]
                if data["cloth_links"]:
                    raise ValueError("separated cloth unexpectedly produced a material force")
            if np.array_equal(frames[-1].state.domains["cloth"].position,
                              world.initial.domains["cloth"].position):
                raise ValueError("separated cloth did not advance")
            checks[spec.key] = dict(kind="separated", accepted_steps=len(frames), force_sites=0)
        elif config.steps == 4:
            checks[spec.key] = certify(world, frames)
        else:
            # Short qualification still checks actual bilateral forces/material points;
            # the four-step nonzero-deformation certificate is not claimed here.
            for frame in frames:
                rows = frame.report.substeps[0].diagnostics["robot_contact"]["cloth_links"]
                if {row["face"] for row in rows} != set(world.keys):
                    raise ValueError("short qualification lost a loaded face")
                for row in rows:
                    for site in row["sites"]:
                        ids, weights = zip(*site["face_weights"], strict=True)
                        lookup = dict(zip(row["face_nodes"], row["face_positions_m"], strict=True))
                        np.testing.assert_allclose(sum(w * lookup[i] for i, w in zip(ids, weights, strict=True)),
                                                   site["point_m"], rtol=0, atol=3e-10)
                        np.testing.assert_array_equal(site["force_on_cloth_n"], -site["force_on_link_n"])
            checks[spec.key] = dict(kind="short qualification", accepted_steps=len(frames))
    return _native(checks)


class ResourceLimit(RuntimeError):
    """A bounded observation stopped; it is not a complete or skipped measurement."""


class _Budget:
    def __init__(self, config):
        self.config, self.start, self.raw_bytes = config, perf_counter_ns(), 0

    def time(self):
        if perf_counter_ns() - self.start > self.config.max_seconds * 1e9:
            raise ResourceLimit("elapsed budget reached at a safe operation boundary")

    def reserve(self, payload):
        data = _encoded(payload) + b"\n"
        if self.raw_bytes + len(data) > self.config.max_raw_bytes:
            raise ResourceLimit("complete-payload JSON budget exceeded; completed windows retained")
        self.raw_bytes += len(data)
        return data


def _measure_window(batch, worlds, config, *, instrumented, cold, budget):
    from phymeo.core.scene import fingerprint

    budget.time()
    batch.reset()
    before, native_before = _counts(batch), _native_counts()
    records, phases = [], []
    step_base = {key: world.initial for key, world in worlds.items()}
    phases_before = batch.last_phases
    probe = _Probe(batch) if instrumented else None
    failure = None
    with ExitStack() as hooks:
        if probe is not None:
            hooks.enter_context(probe)
        begin = perf_counter_ns()
        try:
            for tick in range(config.steps):
                budget.time()
                if probe is not None:
                    probe.step = tick
                phases_before = batch.last_phases
                output = batch.step(1 / 256, {key: world.controls[tick] for key, world in worlds.items()})
                records.append(output)
                phases.extend(batch.last_phases)
                if not all(result.report.accepted for result in output.values()):
                    raise RuntimeError("frozen physical window contains a rejected world")
                step_base = {key: result.state for key, result in output.items()}
        except BaseException as error:
            failure = error
        finally:
            end = perf_counter_ns()
    counts = _delta(_counts(batch), before)
    native = _delta(_native_counts(), native_before)
    raw_frames = [dict(step=i, worlds=[dict(world=k, result=_native(out[k])) for k in worlds])
                  for i, out in enumerate(records)]
    if probe is not None:
        timeline = partition_wall(begin, end, probe.events, caller_thread=probe.main)
        receipts = probe.receipts()
        syncs = probe.syncs
    else:
        timeline = dict(exclusive_ns={"uninstrumented_full_task": end - begin},
                        inclusive_wait_ns={}, spans=[], calls={}, main_thread=get_ident())
        receipts = [_native(phase) | {"step": i // 3} for i, phase in enumerate(phases)]
        # The uninstrumented path does not fabricate a measured sync count.
        syncs = None
    api = Counter()
    extra_api = Counter()
    for label, elapsed in timeline["exclusive_ns"].items():
        if label.startswith("page:"):
            _, _, page, part = label.split(":")
            api[part] += elapsed
            if int(page) > 0:
                extra_api[part] += elapsed
    window = dict(
        whole_ns=end - begin, exclusive_ns=timeline["exclusive_ns"],
        all_page_api_ns=sum(api.values()), extra_page_api_ns=sum(extra_api.values()),
        all_page_api_parts_ns=dict(api), extra_page_api_parts_ns=dict(extra_api),
        inclusive_wait_ns=timeline["inclusive_wait_ns"], raw_timeline=timeline["spans"],
        phase_sites=[p["sites"] for p in receipts], phase_receipts=receipts,
        counter_delta=counts, native_counter_delta=native,
        explicit_synchronizations=syncs, instrumented=instrumented,
        frames=raw_frames, complete_output_sha256=_digest(canonical_frames(raw_frames)),
        accepted_world_steps=sum(result.report.accepted for out in records for result in out.values()),
        counter_audit="incomplete", status="failed" if failure is not None else "complete",
        actual_device=(probe.device if probe is not None and probe.device is not None
                       else _device_signature(batch._layer, config.mode)),
    )
    if failure is not None:
        window["error"] = dict(type=type(failure).__name__, message=str(failure))
        window["failure_transaction"] = dict(
            published_states_sha256=fingerprint(batch.snapshot()),
            last_completed_states_sha256=fingerprint(step_base),
            current_phases_sha256=fingerprint(batch.last_phases),
            phases_before_failed_step_sha256=fingerprint(phases_before),
            completed_step_count=len(records), automatic_retry=False,
        )
        failure.profile_window = window
        raise failure
    try:
        if instrumented:
            actual = dict(counts, explicit_synchronizations=syncs)
            audit = audit_page_counters(window["phase_sites"], actual, mode=config.mode)
            if (probe.completed["kernel_submit"] != counts["completed_launches"]
                    or probe.completed["upload"] != counts["uploads"]
                    or probe.completed["download"] != counts["downloads"]
                    or probe.bytes["upload"] != counts["uploaded_bytes"]
                    or probe.bytes["download"] != counts["downloaded_bytes"]):
                raise ValueError("observed native calls disagree with material counters")
            _check_phase_rows(receipts, worlds, mode=config.mode)
            window["page_audit"] = audit
        else:
            expected_pages = sum((p["sites"] + 127) // 128 for p in receipts)
            window['source_expected_synchronizations'] = expected_pages if config.mode == 'cuda' else 0
            # Validate recorded layer counts independently, but label inferred sync separately.
            audit_page_counters(window["phase_sites"], dict(counts,
                explicit_synchronizations=expected_pages if config.mode == "cuda" else 0), mode=config.mode)
        window["cold_native_extra"] = _audit_native(native, counts, config.mode, cold=cold)
        window["physical_checks"] = _physical_checks(worlds, records, config)
        window["counter_audit"] = "passed"
        window["phase_map"] = _native(phases)
    except BaseException as error:
        window['status'] = 'failed'
        window['error'] = dict(type=type(error).__name__, message=str(error), stage='offline_audit')
        error.profile_window = window
        raise
    return window


def _compact_window(window):
    heavy = {"frames", "raw_timeline", "phase_receipts", "physical_checks", "phase_map", "input_manifest"}
    result = {key: value for key, value in window.items() if key not in heavy}
    result["frame_count"] = len(window.get("frames", ()))
    result["phase_count"] = len(window.get("phase_receipts", ()))
    return result


def run_profile(config, *, cache_root, on_window=None, required_source_root=None):
    """One bounded original-path profile, followed by an independent plain parity run."""
    if not isinstance(config, ProfileConfig):
        raise TypeError("expected frozen ProfileConfig")
    budget = _Budget(config)
    cases = _cases()
    import phymeo as ph

    _check_loaded_sources(required_source_root)
    worlds = cases.build_worlds(config.worlds, config.faces, config.case)
    manifest = _input_manifest(worlds, config)
    manifest_sha = _digest(manifest)
    settings = ph.StepSettings(substeps=1, position_iterations=1,
                               velocity_iterations=1, record_substeps=True)
    cache = Path(cache_root)
    collected = dict(warmups=[], measurements=[])

    def preserve(label, index, window):
        payload = dict(phase=label, index=index, window=window)
        budget.reserve(payload)
        if on_window is not None:
            on_window(label, index, window)
        return _compact_window(window) if on_window is not None else window

    try:
        preserve("input-manifest", 0, dict(input_manifest=manifest, input_manifest_sha256=manifest_sha))
        with ph.TyxMultifaceContactBatch(
            [(key, ph.ClothLinkWorld(world.program, world.initial)) for key, world in worlds.items()],
            config.mode, settings=settings, cache_root=cache / "instrumented",
        ) as batch:
            cold = _measure_window(batch, worlds, config, instrumented=True, cold=True, budget=budget)
            collected["cold"] = preserve("cold", 0, cold)
            for label, count, key in (("warm", config.warm_windows, "warmups"),
                                      ("formal", config.formal_windows, "measurements")):
                for index in range(count):
                    window = _measure_window(batch, worlds, config, instrumented=True, cold=False, budget=budget)
                    collected[key].append(preserve(label, index, window))
            device = _device_signature(batch._layer, config.mode)
        # Rebuild independent programs as well as their workspaces for the plain run.
        reference_worlds = cases.build_worlds(config.worlds, config.faces, config.case)
        if _digest(_input_manifest(reference_worlds, config)) != manifest_sha:
            raise ValueError("plain-reference physical inputs differ from the frozen manifest")
        with ph.TyxMultifaceContactBatch(
            [(key, ph.ClothLinkWorld(world.program, world.initial))
             for key, world in reference_worlds.items()],
            config.mode, settings=settings, cache_root=cache / "plain-reference",
        ) as reference_batch:
            reference = _measure_window(reference_batch, reference_worlds, config,
                                        instrumented=False, cold=True, budget=budget)
            collected["reference"] = preserve("plain-reference", 0, reference)
        expected = collected["reference"]["complete_output_sha256"]
        all_windows = (collected["cold"], *collected["warmups"], *collected["measurements"])
        if any(w["complete_output_sha256"] != expected for w in all_windows):
            raise ValueError("instrumentation changed a complete physical output")
        if _digest(_input_manifest(worlds, config)) != manifest_sha:
            raise ValueError("declared input changed during measurement")
        gate = summarize_windows(config, cold=collected["cold"], warmups=collected["warmups"],
                                 formal=collected["measurements"], complete_output_match=True,
                                 real_cuda=device["real_cuda"])
        budget.time()
        return dict(
            status="complete", config=asdict(config),
            input_manifest=(manifest if on_window is None else dict(
                world_keys=list(worlds), h_s=1 / 256, steps=config.steps, full_payload_in_raw=True)),
            input_manifest_sha256=manifest_sha, complete_output_match=True,
            canonical_normalization="only report.substeps[*].timings values, keys retained",
            actual_device=device, gate=gate, **collected,
            raw_json_bytes=budget.raw_bytes, loaded_modules=_check_loaded_sources(required_source_root),
            execution_scope="TYX normal/friction row law; HOST geometry, mechanics, material mapping and publication",
            timer_scope="instrumented HOST wall; not GPU events or an optimization speed ratio",
            budget_scope="180s ceiling checked at safe boundaries; never interrupts a native call",
        )
    except BaseException as error:
        window = getattr(error, "profile_window", None)
        if window is not None and on_window is not None:
            # Preserve the original failure if its complete raw payload does not fit.
            try:
                preserve("failed-attempt", 0, window)
            except BaseException as archival_error:
                error.add_note(f"failed-attempt archive unavailable: {archival_error}")
        raise


SOURCE_TARGETS = (
    "src/phymeo",
    "scripts/profile_material_phase_transfer.py",
    "tests/scale/material_phase_transfer_cases.py",
    "tests/scale/test_material_phase_transfer_profile.py",
    "tests/scale/multiface_load_cases.py",
    "docs/benchmarks/p65-material-phase-transfer.md",
    "pyproject.toml",
)


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def bind_source(root, expected_head):
    root = Path(root).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", expected_head):
        raise ValueError("expected HEAD must be a full lowercase SHA")
    actual = _git(root, "rev-parse", "HEAD")
    if actual != expected_head:
        raise RuntimeError(f"source HEAD mismatch: expected {expected_head}, got {actual}")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *SOURCE_TARGETS)
    if dirty:
        raise RuntimeError(f"measurement sources must be committed and clean: {dirty}")
    names = _git(root, "ls-files", "--", *SOURCE_TARGETS).splitlines()
    for required in SOURCE_TARGETS[1:]:
        if required not in names:
            raise RuntimeError(f"measurement dependency is not tracked: {required}")
    return dict(root=str(root), head=actual, tracked_paths=names,
                files={name: sha256((root / name).read_bytes()).hexdigest() for name in names})


def verify_source(binding):
    if bind_source(binding["root"], binding["head"]) != binding:
        raise RuntimeError("source changed during the material measurement")


def _sha_file(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RawArchive:
    """New lossless gzip stream, bounded before writing each complete JSON record."""

    def __init__(self, path, *, max_bytes):
        self.path = Path(path)
        self.max_bytes = _integer(max_bytes, "raw JSON budget", 1)
        self.json_bytes, self.records = 0, 0
        self._resources = ExitStack()
        self._stream = None

    def __enter__(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            file = self._resources.enter_context(self.path.open("xb"))
            self._stream = self._resources.enter_context(gzip.GzipFile(fileobj=file, mode="wb", mtime=0))
            return self
        except BaseException:
            self._resources.close()
            raise

    def write(self, value):
        if self._stream is None or self._stream.closed:
            raise RuntimeError("raw archive is closed")
        encoded = _encoded(value) + b"\n"
        if self.json_bytes + len(encoded) > self.max_bytes:
            raise ResourceLimit("raw archive budget exceeded before next complete record")
        self._stream.write(encoded)
        self._stream.flush()
        self.json_bytes += len(encoded)
        self.records += 1

    def __exit__(self, *args):
        self._resources.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cuda", "interpret"), required=True)
    parser.add_argument("--worlds", type=int, required=True)
    parser.add_argument("--faces", type=int, required=True)
    parser.add_argument("--case", choices=("uniform", "heterogeneous"), default="uniform")
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--warm-windows", type=int, default=2)
    parser.add_argument("--formal-windows", type=int, default=3)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--max-raw-bytes", type=int, default=32 * 1024 * 1024)
    args = parser.parse_args(argv)
    args.config = ProfileConfig(
        args.worlds, args.faces, args.mode, args.case, args.warm_windows,
        args.formal_windows, args.steps, args.max_seconds, args.max_raw_bytes,
    )
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_head):
        raise ValueError("expected HEAD must be a full lowercase SHA")
    args.cache_root = args.cache_root or args.out.with_suffix(".cache")
    args.raw_path = args.out.with_suffix(".frames.jsonl.gz")
    paths = (args.out, args.raw_path, args.cache_root)
    if any(not path.is_absolute() for path in paths) or len(set(paths)) != 3:
        raise ValueError("distinct absolute output, sidecar and cache-root paths required")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"fresh measurement path required: {path}")
    return args


def main(argv=None):
    args = parse_args(argv)
    source = bind_source(ROOT, args.expected_head)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=False)
    base = dict(
        protocol="p65-material-phase-transfer-c0", status="running", source=source,
        command=list(sys.argv if argv is None else [str(Path(__file__)), *argv]),
        config=asdict(args.config), output_path=str(args.out), raw_frames_path=str(args.raw_path),
        cache_root=str(args.cache_root), cache_existed_before=False,
        timing_scope="complete four-step output; resets, JSON, file IO and offline audits excluded",
    )
    completed = []
    final, failure, raw = None, None, None
    with args.out.open("x", encoding="utf8", newline="\n") as report:
        try:
            with RawArchive(args.raw_path, max_bytes=args.config.max_raw_bytes) as raw:
                def persist(label, index, window):
                    raw.write(dict(phase=label, index=index, window=window))
                    completed.append(dict(phase=label, index=index, **_compact_window(window)))
                    print(f"{label} {index}: W{args.worlds}, raw JSON {raw.json_bytes} B", flush=True)

                result = run_profile(args.config, cache_root=args.cache_root, on_window=persist,
                                     required_source_root=ROOT)
                verify_source(source)
                final = dict(base, **result, source_verified_after=True)
        except BaseException as error:
            failure = error
            final = dict(base, status="failed", error_type=type(error).__name__, error=str(error),
                         completed_windows=completed, default_changed=False,
                         error_notes=list(getattr(error, '__notes__', ())))
            try:
                verify_source(source)
                final["source_verified_after"] = True
            except BaseException as verification:
                final["source_verified_after"] = False
                final["source_verification_error"] = str(verification)
        finally:
            if args.raw_path.is_file():
                final["raw_frames_sha256"] = _sha_file(args.raw_path)
                final["raw_compressed_bytes"] = args.raw_path.stat().st_size
            if raw is not None:
                final["raw_json_bytes"] = raw.json_bytes
                final["raw_complete_records"] = raw.records
            final["storage_cap"] = dict(raw_json_bytes=args.config.max_raw_bytes,
                                        summary_json_bytes=MAX_SUMMARY_BYTES,
                                        compression="gzip lossless, complete records")
            encoded = _encoded(final)
            if len(encoded) > MAX_SUMMARY_BYTES:
                failure = ResourceLimit("summary JSON budget exceeded; complete raw sidecar retained")
                final = dict(status="failed", error_type=type(failure).__name__, error=str(failure),
                             raw_frames_path=str(args.raw_path), source_head=source["head"],
                             raw_frames_sha256=_sha_file(args.raw_path) if args.raw_path.exists() else None)
                encoded = _encoded(final)
            report.write(encoded.decode("utf8") + "\n")
            report.flush()
    if failure is not None:
        raise failure
    print(json.dumps(dict(output=str(args.out), actual_device=final["actual_device"],
                          gate=final["gate"], status=final["status"]), allow_nan=False))


if __name__ == "__main__":
    main()


