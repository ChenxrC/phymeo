"""Frozen K0 query identities; tests first, AUTHORING ONLY / UNEXECUTED.

The existing public model and HOST literal geometry precede the first missing
public query capability. No new-product import runs during collection.
"""

import gzip
import hashlib
import importlib.util
import json
import math
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph

FIRST_FOUR = ("SS-sep", "SS-overlap", "CH-sep", "CH-overlap")
Q01 = FIRST_FOUR
RAW_SHAPES = {"meta": (16,), "gap": (4,), "normal": (3,), "surface": (2, 3), "material_ids": (2, 4), "barycentric": (2, 4), "skin_offset": (2, 3)}
STATS = ("allocations", "live_bytes", "peak_owned_bytes", "static_uploads", "dynamic_uploads", "uploaded_bytes", "downloads", "downloaded_bytes", "launches", "completed_launches", "synchronizations", "attempts", "queries", "failed_queries", "pose_gather_launches", "query_launches", "pair_uploads")


@lru_cache(maxsize=1)
def reference():
    path = Path(__file__).resolve().parents[1] / "reference_newton" / "rr" / "device_convex_query_reference.py"
    spec = importlib.util.spec_from_file_location("k0_independent_finite_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode(value):
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape), "bytes_hex": value.tobytes(order="C").hex(), "values": encode(value.tolist())}
    if isinstance(value, np.generic):
        return encode(value.item())
    if type(value) is object:
        return {"opaque_process_local_type": "builtins.object", "identity_serialized": False}
    if is_dataclass(value):
        return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: encode(item) for key, item in value.items()}
        return {"mapping_entries": [[encode(key), encode(item)] for key, item in value.items()]}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unrecorded type {type(value).__module__}.{type(value).__qualname__}")


def canonical(value):
    return json.dumps(encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


@contextmanager
def saved(tmp_path, identity, mode, inputs):
    sources = [Path(__file__).resolve(), Path(reference().__file__).resolve()]
    record = {"schema": "K0-query-evidence-v1", "identity": identity, "mode": mode, "inputs": inputs, "source": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes_hex": p.read_bytes().hex()} for p in sources], "attempts": [], "completed": False, "failure": None}
    try:
        yield record
        record["completed"] = True
    except BaseException as error:
        record["failure"] = {"type": type(error).__name__, "message": str(error), "stage": record.get("stage")}
        raise
    finally:
        target = tmp_path / f"{identity}-{mode}.json.gz"
        with gzip.open(target, "wt", encoding="utf-8") as stream:
            json.dump(encode(record), stream, sort_keys=True, allow_nan=False)


def ordinary_forest(n=1, *, q=None, velocity=None, anchor=(0, 0, 0)):
    positions = np.tile(np.asarray(anchor, dtype=float), (n, 1))
    model = ph.RobotModel(np.full(n, -1, dtype=np.int32), ("prismatic",) * n, positions, np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)), np.tile([0.0, 0.0, 1.0], (n, 1)), np.ones(n), np.zeros((n, 3)), np.full((n, 3), 1 / 16), np.zeros(3))
    program = ph.DrivenRobotProgram(model, kp=np.zeros(n), kd=np.zeros(n), effort_limit=np.full(n, 8.0), delay_steps=0)
    state = program.initial_state(ph.RobotState(np.zeros(n) if q is None else np.asarray(q, dtype=float), np.zeros(n) if velocity is None else np.asarray(velocity, dtype=float)))
    return ph.FixedRobotTree(program, state)


def registrations(case, topology):
    shapes, bindings = [], []
    for side in range(2):
        key = ph.CollisionKey(*case["keys"][side])
        kind = {"S": "sphere", "C": "capsule", "B": "hull", "H": "hull"}[case["kinds"][side]]
        shape = ph.CollisionShape(key, kind, tuple(case["ids"][side]), radius=case["radii"][side], revision=7 + side)
        shapes.append(shape)
        if side == 0:
            bindings.append(ph.ResidentTreeBinding(key, key.world, 0, case["points"][side], topology=topology))
        else:
            bindings.append(ph.ResidentStaticBinding(key, ph.CollisionMotion(case["points"][side])))
    if case["registration_reverse"]:
        shapes.reverse()
        bindings.reverse()
    return ph.CollisionModel(tuple(shapes)), tuple(bindings)


def query_attempt(batch, plan, record, label):
    item = {"label": label, "returned": False, "result": None, "error": None, "stats_before": dict(plan.stats)}
    record["attempts"].append(item)
    try:
        result = batch.convex_query(plan)
        item.update(returned=True, result=result)
        return result
    except BaseException as error:
        item["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        item["stats_after"] = dict(plan.stats)


def check_raw(result):
    assert isinstance(result.raw, Mapping) and set(RAW_SHAPES) <= set(result.raw)
    n = len(result.rows)
    for name, tail in RAW_SHAPES.items():
        value = result.raw[name]
        assert isinstance(value, np.ndarray) and value.shape[1:] == tail and len(value) >= n
        assert value.dtype == np.dtype(np.int32 if name in ("meta", "material_ids") else np.float32)
        assert not value.flags.writeable
        if value.dtype.kind == "f":
            assert np.isfinite(value).all()
    meta = result.raw["meta"][:n]
    assert np.all((meta[:, 0] >= 1) & (meta[:, 0] <= 4))
    assert np.all(meta[:, 14:] == 0)
    assert np.all(meta[:, 12] == result.attempt_tag)
    np.testing.assert_array_equal(meta[:, 13], np.arange(n))
    assert isinstance(result.source, Mapping) and result.source
    assert set(STATS) <= set(result.stats) or {"attempt_delta", "broadphase_delta", "logical_pairs", "padded_pairs"} <= set(result.stats)


def check_parallel_row(row, case):
    ref = reference()
    first_key, second_key = (ph.CollisionKey(*key) for key in case["keys"])
    expected_keys = tuple(sorted((first_key, second_key)))
    assert row.pair == expected_keys
    flip = expected_keys[0] != first_key
    sides = (1, 0) if flip else (0, 1)
    normal = -case["normal"] if flip else case["normal"]
    length = ref.scale(case)
    assert abs(row.scale_m - length) <= 1e-4 * length
    if abs(case["gap_m"]) <= 2e-4 * length:
        assert row.relation != "SEPARATED"
        if row.status != "RESOLVED":
            return {"unresolved": True, "reason": row.reason}
    assert row.status == "RESOLVED" and row.gap_valid and row.normal_valid and row.witness_valid
    assert row.first.key == expected_keys[0] and row.second.key == expected_keys[1]
    assert abs(row.gap_m - case["gap_m"]) <= 1e-4 * length
    assert row.gap_lower_m <= row.gap_m <= row.gap_upper_m
    assert np.linalg.norm(np.asarray(row.normal) - normal) <= 2e-3
    assert abs(np.linalg.norm(row.normal) - 1.0) <= 2e-4
    if abs(case["gap_m"]) > 2e-4 * length:
        assert row.relation == ("SEPARATED" if case["gap_m"] > 0 else "OVERLAP")
    origin = np.asarray(row.origin_m, dtype=float)
    checks = []
    for output_side, original_side in enumerate(sides):
        witness = row.first if output_side == 0 else row.second
        checks.append(ref.finite_material_check(np.asarray(case["points"][original_side]) - origin, case["radii"][original_side], case["ids"][original_side], witness, normal, output_side, tolerance_m=2e-4 * length))
    delta = np.asarray(row.second.surface_relative_m) - row.first.surface_relative_m
    assert np.linalg.norm(delta - row.gap_m * np.asarray(row.normal)) <= 2e-4 * length
    return {"declared_gap_m": case["gap_m"], "actual_gap_m": row.gap_m, "scale_from_input_m": length, "material": checks}


def run_parallel(case, identity, tyx_mode, tmp_path):
    ref = reference()
    with saved(tmp_path, identity, tyx_mode, case) as record:
        record["stage"] = "existing_literal_geometry"
        record["literal_oracle"] = ref.literal_parallel_precondition(case)
        # Existing HOST finite material geometry is a premise, never the sole oracle.
        from phymeo.geometry.contact_patch_adapter import contact_patch_sample
        keys = tuple(ph.CollisionKey(*key) for key in case["keys"])
        premise_shapes = tuple(ph.CollisionShape(key, {"S": "sphere", "C": "capsule", "B": "hull", "H": "hull"}[kind], ids, radius=radius) for key, kind, ids, radius in zip(keys, case["kinds"], case["ids"], case["radii"], strict=True))
        patch = contact_patch_sample(*premise_shapes, *case["points"])
        assert patch.sites
        for site in patch.sites:
            assert abs(site.witness.gap_m - case["gap_m"]) <= 1e-10
        record["host_premise"] = patch
        world = keys[0].world
        with ph.TyxWorldBatch([(world, ordinary_forest())], tyx_mode, cache_root=tmp_path) as batch:
            before = batch.snapshot()
            record["before"] = before
            model, bindings = registrations(case, before[world].topology)
            record.update(model=model, bindings=bindings)
            with batch.prepare_collision(model, bindings) as broad:
                record["stage"] = "new_public_prepare_convex_query"
                requested = (tuple(reversed(keys)) if case["requested_reverse"] else keys,)
                plan = batch.prepare_convex_query(broad, pairs=requested)
                try:
                    record["stage"] = "query"
                    result = query_attempt(batch, plan, record, "only_query")
                    assert plan.last_result is result and plan.require_current(result) is result
                    check_raw(result)
                    assert len(result.rows) == 1 and result.pair_source == "explicit"
                    assert result.requested_pairs == requested and tuple(result.request_row) == (0,)
                    assert tuple(result.request_reversed) == (requested[0] != tuple(sorted(keys)),)
                    assert tuple(result.world_keys) == (world,) and tuple(result.ticks) == (0,)
                    record["checks"] = check_parallel_row(result.rows[0], case)
                    assert canonical(batch.snapshot()) == canonical(before)
                    record["after"] = batch.snapshot()
                    record["result_owned"] = encode(result)
                    assert set(STATS) <= set(plan.stats)
                finally:
                    plan.close()
                    record["closed_query_stats"] = dict(plan.stats)
                    assert plan.stats["live_bytes"] == 0
            record["broad_stats"] = broad.stats()
        record["closed_batch_stats"] = batch.stats()


@pytest.mark.tyx
@pytest.mark.parametrize("identity", Q01, ids=lambda identity: f"K0-Q01-{identity}")
def test_k0_family_gap_witness(identity, tyx_mode, tmp_path):
    family, phase = identity.split("-")
    case = reference().family_case(family, reference().GAPS[phase])
    run_parallel(case, f"K0-Q01-{identity}", tyx_mode, tmp_path)