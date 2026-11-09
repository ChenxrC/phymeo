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
from contextlib import closing, contextmanager
from dataclasses import fields, is_dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph

FIRST_FOUR = ("SS-sep", "SS-overlap", "CH-sep", "CH-overlap")
Q01 = FIRST_FOUR + tuple(
    f"{family}-{phase}"
    for family in ("SS", "SC", "SB", "SH", "CC", "CB", "CH", "BB", "BH", "HH")
    for phase in ("sep", "touch", "overlap")
    if f"{family}-{phase}" not in FIRST_FOUR
)
RAW_SHAPES = {
    "meta": (16,),
    "gap": (4,),
    "normal": (3,),
    "surface": (2, 3),
    "material_ids": (2, 4),
    "barycentric": (2, 4),
    "skin_offset": (2, 3),
}
STATS = (
    "allocations",
    "live_bytes",
    "peak_owned_bytes",
    "static_uploads",
    "dynamic_uploads",
    "uploaded_bytes",
    "downloads",
    "downloaded_bytes",
    "launches",
    "completed_launches",
    "synchronizations",
    "attempts",
    "queries",
    "failed_queries",
    "pose_gather_launches",
    "query_launches",
    "pair_uploads",
)


@lru_cache(maxsize=1)
def reference():
    path = (
        Path(__file__).resolve().parents[1]
        / "reference_newton"
        / "rr"
        / "device_convex_query_reference.py"
    )
    spec = importlib.util.spec_from_file_location("k0_independent_finite_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode(value):
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "bytes_hex": value.tobytes(order="C").hex(),
            "values": encode(value.tolist()),
        }
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, np.dtype):
        return {"numpy_dtype": value.str}
    if isinstance(value, type) and value in (np.float32, np.float64, np.int32, np.int64):
        return {"numpy_dtype": np.dtype(value).str}
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
    record = {
        "schema": "K0-query-evidence-v1",
        "identity": identity,
        "mode": mode,
        "inputs": inputs,
        "source": [
            {
                "path": str(p),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes_hex": p.read_bytes().hex(),
            }
            for p in sources
        ],
        "attempts": [],
        "completed": False,
        "failure": None,
    }
    try:
        yield record
        record["completed"] = True
    except BaseException as error:
        record["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "stage": record.get("stage"),
        }
        raise
    finally:
        target = tmp_path / f"{identity}-{mode}.json.gz"
        with gzip.open(target, "wt", encoding="utf-8") as stream:
            json.dump(encode(record), stream, sort_keys=True, allow_nan=False)


def ordinary_forest(n=1, *, q=None, velocity=None, anchor=(0, 0, 0), orientation=(0, 0, 0, 1)):
    positions = np.tile(np.asarray(anchor, dtype=float), (n, 1))
    model = ph.RobotModel(
        np.full(n, -1, dtype=np.int32),
        ("prismatic",) * n,
        positions,
        np.tile(np.asarray(orientation, dtype=float), (n, 1)),
        np.tile([0.0, 0.0, 1.0], (n, 1)),
        np.ones(n),
        np.zeros((n, 3)),
        np.full((n, 3), 1 / 16),
        np.zeros(3),
    )
    program = ph.DrivenRobotProgram(
        model, kp=np.zeros(n), kd=np.zeros(n), effort_limit=np.full(n, 8.0), delay_steps=0
    )
    state = program.initial_state(
        ph.RobotState(
            np.zeros(n) if q is None else np.asarray(q, dtype=float),
            np.zeros(n) if velocity is None else np.asarray(velocity, dtype=float),
        )
    )
    return ph.FixedRobotTree(program, state)


def registrations(case, topology):
    shapes, bindings = [], []
    for side in range(2):
        key = ph.CollisionKey(*case["keys"][side])
        kind = {"S": "sphere", "C": "capsule", "B": "hull", "H": "hull"}[case["kinds"][side]]
        shape = ph.CollisionShape(
            key, kind, tuple(case["ids"][side]), radius=case["radii"][side], revision=7 + side
        )
        shapes.append(shape)
        if side == 0:
            bindings.append(
                ph.ResidentTreeBinding(key, key.world, 0, case["local"][side], topology=topology)
            )
        else:
            bindings.append(ph.ResidentStaticBinding(key, ph.CollisionMotion(case["points"][side])))
    if case["registration_reverse"]:
        shapes.reverse()
        bindings.reverse()
    return ph.CollisionModel(tuple(shapes)), tuple(bindings)


def query_attempt(batch, plan, record, label):
    item = {
        "label": label,
        "returned": False,
        "result": None,
        "error": None,
        "stats_before": dict(plan.stats),
    }
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
    for name, value in result.raw.items():
        assert np.all(value[n:] == (-1 if name == "material_ids" else 0))
    assert np.all((meta[:, 0] >= 1) & (meta[:, 0] <= 4))
    assert np.all(meta[:, 14:] == 0)
    assert np.all(meta[:, 12] == result.attempt_tag)
    np.testing.assert_array_equal(meta[:, 13], np.arange(n))
    assert isinstance(result.source, Mapping) and result.source
    assert {"attempt_delta", "broadphase_delta", "logical_pairs", "padded_pairs"} <= set(
        result.stats
    )
    assert result.stats["logical_pairs"] == n and result.stats["padded_pairs"] == len(
        result.raw["meta"]
    )
    check_wire_rows(result)


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
        checks.append(
            ref.finite_material_check(
                np.asarray(case["points"][original_side]) - origin,
                case["radii"][original_side],
                case["ids"][original_side],
                witness,
                normal,
                output_side,
                tolerance_m=2e-4 * length,
            )
        )
    delta = np.asarray(row.second.surface_relative_m) - row.first.surface_relative_m
    assert np.linalg.norm(delta - row.gap_m * np.asarray(row.normal)) <= 2e-4 * length
    return {
        "declared_gap_m": case["gap_m"],
        "actual_gap_m": row.gap_m,
        "scale_from_input_m": length,
        "material": checks,
    }


def run_parallel(case, identity, tyx_mode, tmp_path):
    ref = reference()
    with saved(tmp_path, identity, tyx_mode, case) as record:
        record["stage"] = "existing_literal_geometry"
        record["literal_oracle"] = ref.literal_parallel_precondition(case)
        # Existing HOST finite material geometry is a premise, never the sole oracle.
        from phymeo.geometry.contact_patch_adapter import contact_patch_sample

        keys = tuple(ph.CollisionKey(*key) for key in case["keys"])
        premise_shapes = tuple(
            ph.CollisionShape(
                key,
                {"S": "sphere", "C": "capsule", "B": "hull", "H": "hull"}[kind],
                ids,
                radius=radius,
            )
            for key, kind, ids, radius in zip(
                keys, case["kinds"], case["ids"], case["radii"], strict=True
            )
        )
        patch = contact_patch_sample(*premise_shapes, *case["points"])
        assert patch.sites
        for site in patch.sites:
            assert abs(site.witness.gap_m - case["gap_m"]) <= 1e-10
        record["host_premise"] = patch
        world = keys[0].world
        with ph.TyxWorldBatch(
            [
                (
                    world,
                    ordinary_forest(anchor=case["positions"][0], orientation=case_quaternion(case)),
                )
            ],
            tyx_mode,
            cache_root=tmp_path,
        ) as batch:
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
                    check_source(result, model, bindings, before, case=case)
                    record["checks"] = check_case_row(result.rows[0], case, result.source)
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


def case_quaternion(case):
    variant = case.get("transform", {}).get("variant")
    if variant in ("cycle", "cycle_reverse_ids"):
        return (0.5, 0.5, 0.5, 0.5)
    if variant == "tilt":
        return (0.0, 1 / math.sqrt(5), 0.0, 2 / math.sqrt(5))
    return (0.0, 0.0, 0.0, 1.0)


def check_wire_rows(result):
    status = ("UNWRITTEN", "RESOLVED", "UNKNOWN", "UNSUPPORTED", "RESOURCE_LIMIT")
    relation = ("UNDETERMINED", "SEPARATED", "OVERLAP", "CONTACT_BAND")
    reason = (
        "OK",
        "NEAR_ZERO",
        "NORMAL_NONUNIQUE",
        "CORE_CONTAINMENT",
        "GJK_LIMIT",
        "EPA_LIMIT",
        "DEGENERATE_SIMPLEX",
        "EPA_CAPACITY",
        "UNSUPPORTED_SHAPE",
        "SOURCE_PRECISION",
        "STALLED",
    )
    phase = ("NONE", "ANALYTIC_ROUND", "GJK_CORE", "GJK_POLYTOPE", "EPA")
    for index, row in enumerate(result.rows):
        meta = result.raw["meta"][index]
        assert (
            0 <= int(meta[1]) < len(relation)
            and 0 <= int(meta[2]) < len(reason)
            and 0 <= int(meta[3]) < len(phase)
        )
        assert np.all(meta[4:9] >= 0)
        assert row.status == status[int(meta[0])]
        assert row.relation == relation[int(meta[1])]
        assert row.reason == reason[int(meta[2])]
        assert row.phase == phase[int(meta[3])]
        assert 0 <= int(meta[9]) <= 7
        assert (row.gap_valid, row.normal_valid, row.witness_valid) == tuple(
            bool(meta[9] & flag) for flag in (1, 2, 4)
        )
        assert tuple(float(x) for x in result.raw["gap"][index]) == (
            row.gap_m,
            row.gap_lower_m,
            row.gap_upper_m,
            row.scale_m,
        )
        np.testing.assert_array_equal(row.normal, result.raw["normal"][index])
        assert tuple(int(x) for x in meta[4:9]) == (
            row.gjk_used,
            row.epa_used,
            row.support_calls,
            row.vertices_used,
            row.faces_used,
        )
        assert (
            row.gjk_used <= 64
            and row.epa_used <= 192
            and row.vertices_used <= 64
            and row.faces_used <= 128
        )
        for side, witness in enumerate((row.first, row.second)):
            count = int(meta[10 + side])
            assert 0 <= count <= 4 and witness.key == row.pair[side]
            assert np.all(result.raw["material_ids"][index, side, count:] == -1)
            assert np.all(result.raw["barycentric"][index, side, count:] == 0)
            if not row.witness_valid:
                assert np.all(result.raw["surface"][index, side] == 0) and np.all(
                    result.raw["skin_offset"][index, side] == 0
                )
            assert witness.valid == row.witness_valid
            assert tuple(int(x) for x in result.raw["material_ids"][index, side, :count]) == tuple(
                witness.vertex_ids
            )
            assert tuple(float(x) for x in result.raw["barycentric"][index, side, :count]) == tuple(
                witness.weights
            )
            np.testing.assert_array_equal(witness.offset_m, result.raw["skin_offset"][index, side])
            np.testing.assert_array_equal(
                witness.surface_relative_m, result.raw["surface"][index, side]
            )
        if not row.gap_valid:
            assert (row.gap_m, row.gap_lower_m, row.gap_upper_m) == (0, 0, 0)
        if not row.normal_valid:
            assert np.all(result.raw["normal"][index] == 0)
        if row.status == "RESOLVED":
            assert row.gap_valid and row.normal_valid and row.witness_valid
    for value in result.raw.values():
        with pytest.raises(ValueError):
            value.setflags(write=True)


def check_source(result, model, bindings, checkpoint, *, case=None):
    src = result.source
    required = {
        "schema",
        "model",
        "bindings",
        "world_keys",
        "world_origins_m",
        "shape_world_rows",
        "shape_keys",
        "shape_revisions",
        "shape_kinds",
        "shape_rows",
        "shape_groups",
        "vertex_offsets",
        "vertex_ids",
        "declared_local_points_m",
        "uploaded_local_points_m",
        "upload_frame",
        "point_quantization_error_m",
        "declared_radius_m",
        "uploaded_radius_m",
        "radius_quantization_error_m",
        "pose",
        "stamp",
        "allowed_pairs",
    }
    assert required <= set(src) and src["schema"] == "resident-convex-source-v1"
    shapes = model.shapes
    ordered = tuple({binding.key: binding for binding in bindings}[shape.key] for shape in shapes)
    assert src["model"] == model and canonical(src["bindings"]) == canonical(ordered)
    assert (
        tuple(src["shape_keys"]) == tuple(shape.key for shape in shapes) == tuple(result.shape_keys)
    )
    assert (
        tuple(src["shape_revisions"])
        == tuple(shape.revision for shape in shapes)
        == tuple(result.shape_revisions)
    )
    assert tuple(src["shape_kinds"]) == tuple(shape.kind for shape in shapes)
    assert (
        tuple(src["world_keys"])
        == tuple(sorted({shape.key.world for shape in shapes}))
        == tuple(result.world_keys)
    )
    n = len(shapes)
    v = sum(len(shape.vertex_ids) for shape in shapes)
    arrays = {
        "world_origins_m": (np.float64, (len(src["world_keys"]), 3)),
        "shape_world_rows": (np.int32, (n,)),
        "shape_rows": (np.int32, (n,)),
        "shape_groups": (np.int32, (n,)),
        "vertex_offsets": (np.int32, (n + 1,)),
        "vertex_ids": (np.int32, (v,)),
        "declared_local_points_m": (np.float64, (v, 3)),
        "uploaded_local_points_m": (np.float32, (v, 3)),
        "point_quantization_error_m": (np.float64, (v, 3)),
        "declared_radius_m": (np.float64, (n,)),
        "uploaded_radius_m": (np.float32, (n,)),
        "radius_quantization_error_m": (np.float64, (n,)),
        "pose": (np.float32, (n, 12)),
        "stamp": (np.int32, (n, 3)),
    }
    for name, (dtype, shape) in arrays.items():
        array = src[name]
        assert (
            isinstance(array, np.ndarray)
            and array.dtype == np.dtype(dtype)
            and array.shape == shape
        )
        assert not array.flags.writeable and np.isfinite(array).all()
        with pytest.raises(ValueError):
            array.setflags(write=True)
    expected_offsets = np.asarray(
        [0, *np.cumsum([len(shape.vertex_ids) for shape in shapes])], dtype=np.int32
    )
    np.testing.assert_array_equal(src["vertex_offsets"], expected_offsets)
    np.testing.assert_array_equal(
        src["vertex_ids"], [node for shape in shapes for node in shape.vertex_ids]
    )
    worlds = tuple(checkpoint)
    for row, (shape, binding) in enumerate(zip(shapes, ordered, strict=True)):
        start, stop = expected_offsets[row : row + 2]
        world_row = src["world_keys"].index(shape.key.world)
        assert src["shape_world_rows"][row] == world_row
        np.testing.assert_array_equal(
            src["stamp"][row], (1, checkpoint[shape.key.world].tick, worlds.index(shape.key.world))
        )
        if isinstance(binding, ph.ResidentStaticBinding):
            declared = binding.motion.start
            frame = "world_anchor_relative"
            uploaded_base = declared - src["world_origins_m"][world_row]
            assert src["shape_rows"][row] == src["shape_groups"][row] == -1
            np.testing.assert_array_equal(src["pose"][row], [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
        else:
            declared = binding.local_points_m
            frame = "body_local"
            uploaded_base = declared
            assert src["shape_groups"][row] == binding.group_index and src["shape_rows"][row] >= 0
        assert src["upload_frame"][row] == frame
        np.testing.assert_array_equal(src["declared_local_points_m"][start:stop], declared)
        uploaded = np.asarray(uploaded_base, dtype=np.float32)
        np.testing.assert_array_equal(src["uploaded_local_points_m"][start:stop], uploaded)
        np.testing.assert_array_equal(
            src["point_quantization_error_m"][start:stop], uploaded.astype(float) - uploaded_base
        )
    np.testing.assert_array_equal(src["declared_radius_m"], [shape.radius for shape in shapes])
    np.testing.assert_array_equal(
        src["uploaded_radius_m"], np.array([shape.radius for shape in shapes], np.float32)
    )
    np.testing.assert_array_equal(
        src["radius_quantization_error_m"],
        src["uploaded_radius_m"].astype(float) - src["declared_radius_m"],
    )
    if case is not None:
        np.testing.assert_array_equal(
            src["world_origins_m"],
            np.asarray([case["positions"][0]], dtype=np.float32).astype(float),
        )
        first_row = src["shape_keys"].index(ph.CollisionKey(*case["keys"][0]))
        pose = src["pose"][first_row].astype(float)
        origin = src["world_origins_m"][src["shape_world_rows"][first_row]]
        tolerance = 2e-4 * reference().scale(case)
        assert np.linalg.norm(pose[:3] + origin - case["positions"][0]) <= tolerance
        assert np.linalg.norm(pose[3:].reshape(3, 3) - case["rotations"][0]) <= 2e-4


def check_case_row(row, case, source):
    checked = check_parallel_row(row, case)
    actual = reference().stored_oracle(case, source)
    checked["actual_uploaded_oracle"] = actual
    checked["declared_oracle"] = reference().finite_oracle(
        case["kinds"], case["points"], case["radii"]
    )
    if row.status == "RESOLVED":
        assert actual["gap_m"] is not None
        tolerance = 1e-4 * reference().scale(case)
        assert abs(row.gap_m - actual["gap_m"]) <= tolerance
        if abs(actual["gap_m"]) > 2e-4 * reference().scale(case):
            sign = -1 if row.pair[0] != ph.CollisionKey(*case["keys"][0]) else 1
            assert np.linalg.norm(np.asarray(row.normal) - sign * actual["normal"]) <= 2e-3
    return checked


FINITE_IDS = (
    "end_box",
    "cap_side",
    "cap_end",
    "cap_oblique",
    "cc_parallel",
    "cc_skew",
    "cc_end_end",
    "h_slant",
    "h_vertex",
)
FRAME_IDS = tuple(
    (family, variant)
    for family in ("SS", "SC", "SB", "SH", "CC", "CB", "CH", "BB", "BH", "HH")
    for variant in ("reverse", "cycle", "tilt", "permute", "cycle_reverse_ids")
)
NEAR_IDS = tuple(
    (family, label, value)
    for family in ("SS", "CH", "HH")
    for label, value in (
        ("pos-far", 2**-10),
        ("neg-far", -(2**-10)),
        ("pos-near", 2**-20),
        ("neg-near", -(2**-20)),
    )
)


@pytest.mark.tyx
@pytest.mark.parametrize("identity", FINITE_IDS, ids=lambda identity: f"K0-Q02-{identity}")
def test_k0_finite_features(identity, tyx_mode, tmp_path):
    case = reference().finite_case(identity)
    oracle = reference().finite_oracle(case["kinds"], case["points"], case["radii"])
    assert abs(oracle["gap_m"] - case["gap_m"]) <= 1e-12
    assert np.linalg.norm(oracle["normal"] - case["normal"]) <= 1e-12
    run_query_case(case, f"K0-Q02-{identity}", tyx_mode, tmp_path, oracle=oracle)


@pytest.mark.tyx
@pytest.mark.parametrize(
    "family,variant", FRAME_IDS, ids=[f"K0-Q03-{family}-{variant}" for family, variant in FRAME_IDS]
)
def test_k0_frame_and_identity(family, variant, tyx_mode, tmp_path):
    case = reference().transformed(reference().family_case(family, -0.01), variant)
    run_parallel(case, f"K0-Q03-{family}-{variant}", tyx_mode, tmp_path)


@pytest.mark.tyx
@pytest.mark.parametrize(
    "family,label,gap",
    NEAR_IDS,
    ids=[f"K0-Q06-band-{family}-{label}" for family, label, _ in NEAR_IDS],
)
def test_k0_near_zero(family, label, gap, tyx_mode, tmp_path):
    case = reference().family_case(family, gap)
    run_parallel(case, f"K0-Q06-band-{family}-{label}", tyx_mode, tmp_path)


def run_query_case(case, identity, tyx_mode, tmp_path, *, oracle, expected=None):
    with saved(tmp_path, identity, tyx_mode, case) as record:
        record["independent_oracle"] = oracle
        world = case["keys"][0][0]
        with ph.TyxWorldBatch(
            [
                (
                    world,
                    ordinary_forest(anchor=case["positions"][0], orientation=case_quaternion(case)),
                )
            ],
            tyx_mode,
            cache_root=tmp_path,
        ) as batch:
            before = batch.snapshot()
            record["before"] = before
            record["stage"] = "registration"
            model, bindings = registrations(case, before[world].topology)
            record.update(model=model, bindings=bindings)
            with batch.prepare_collision(model, bindings) as broad:
                record["stage"] = "query_plan"
                pairs = (tuple(ph.CollisionKey(*key) for key in case["keys"]),)
                with closing(batch.prepare_convex_query(broad, pairs=pairs)) as plan:
                    record["stage"] = "query"
                    result = query_attempt(batch, plan, record, "only_query")
                    check_raw(result)
                    check_source(result, model, bindings, before, case=case)
                    assert len(result.rows) == 1 and result.requested_pairs == pairs
                    if expected is None:
                        record["checks"] = check_case_row(result.rows[0], case, result.source)
                    else:
                        expected(result.rows[0], record)
                    assert canonical(batch.snapshot()) == canonical(before)
                    record["after"] = batch.snapshot()
                record["closed_query_stats"] = dict(plan.stats)
                assert plan.stats["live_bytes"] == 0
            record["broad_stats"] = broad.stats()
        record["closed_batch_stats"] = batch.stats()


DOMAIN_IDS = (
    "coincident_ss",
    "deep_center",
    "deep_offset",
    "identical_hh",
    "zero_axis",
    "flat_h_far",
    "duplicate_xyz_far",
    "bad_address-repeat_id",
    "bad_address-missing_point",
    "nonfinite-nan",
    "nonfinite-inf",
    "rounded_h",
    "h_capacity-v64",
    "h_capacity-v65",
    "origin-rebase",
    "origin-loss",
)
INVALID_SOURCE = {
    "flat_h_far",
    "duplicate_xyz_far",
    "bad_address-repeat_id",
    "bad_address-missing_point",
    "nonfinite-nan",
    "nonfinite-inf",
    "h_capacity-v65",
}


@pytest.mark.tyx
@pytest.mark.parametrize("identity", DOMAIN_IDS, ids=lambda identity: f"K0-Q06-domain-{identity}")
def test_k0_domain_classification(identity, tyx_mode, tmp_path):
    case = reference().domain_case(identity)
    with saved(tmp_path, f"K0-Q06-domain-{identity}", tyx_mode, case) as record:
        world = case["keys"][0][0]
        with ph.TyxWorldBatch(
            [(world, ordinary_forest(anchor=case["positions"][0]))], tyx_mode, cache_root=tmp_path
        ) as batch:
            before = batch.snapshot()
            record["before"] = before
            broad = plan = None
            try:
                record["stage"] = "original_registration"
                model, bindings = registrations(case, before[world].topology)
                record.update(model=model, bindings=bindings)
                broad = batch.prepare_collision(model, bindings)
                record["stage"] = "new_query_preflight"
                pairs = (tuple(ph.CollisionKey(*key) for key in case["keys"]),)
                plan = batch.prepare_convex_query(broad, pairs=pairs)
                record["stage"] = "device_query"
                result = query_attempt(batch, plan, record, "only_query")
                assert identity not in INVALID_SOURCE, "illegal source was accepted as a query"
                check_raw(result)
                check_source(
                    result,
                    model,
                    bindings,
                    before,
                    case=None if identity == "origin-loss" else case,
                )
                assert len(result.rows) == 1
                row = result.rows[0]
                if identity in ("zero_axis", "h_capacity-v64"):
                    record["checks"] = check_case_row(row, case, result.source)
                    if identity == "zero_axis":
                        first = row.first if row.first.key == pairs[0][0] else row.second
                        assert dict(zip(first.vertex_ids, first.weights, strict=True)) == {
                            100: 0.5,
                            101: 0.5,
                        }
                elif identity == "origin-rebase":
                    if row.status == "RESOLVED":
                        record["checks"] = check_case_row(row, case, result.source)
                    else:
                        assert row.reason == "SOURCE_PRECISION" and not row.witness_valid
                elif identity == "origin-loss":
                    assert (
                        row.status in ("UNKNOWN", "UNSUPPORTED")
                        and row.reason == "SOURCE_PRECISION"
                    )
                    assert not row.witness_valid and row.relation != "SEPARATED"
                elif identity == "rounded_h":
                    assert row.status == "UNSUPPORTED" and row.reason == "UNSUPPORTED_SHAPE"
                    assert not row.witness_valid
                    assert 0.01 in tuple(result.source["declared_radius_m"])
                else:
                    assert row.status in ("UNKNOWN", "UNSUPPORTED", "RESOURCE_LIMIT")
                    assert row.relation in ("OVERLAP", "UNDETERMINED") and not row.normal_valid
                    assert row.reason in (
                        "NORMAL_NONUNIQUE",
                        "CORE_CONTAINMENT",
                        "DEGENERATE_SIMPLEX",
                        "EPA_LIMIT",
                        "EPA_CAPACITY",
                        "STALLED",
                    )
                record["classification"] = row
            except (TypeError, ValueError, RuntimeError) as error:
                if identity not in INVALID_SOURCE | {"origin-loss", "origin-rebase"}:
                    raise
                patterns = {
                    "flat_h_far": ("dimensional", "full_rank", "rank", "convex"),
                    "duplicate_xyz_far": ("duplicate",),
                    "bad_address-repeat_id": ("vertex", "id"),
                    "bad_address-missing_point": ("vertex", "material", "point"),
                    "nonfinite-nan": ("finite",),
                    "nonfinite-inf": ("finite",),
                    "h_capacity-v65": ("capacity", "64"),
                    "origin-loss": (
                        "precision",
                        "represent",
                        "resolution",
                        "duplicate",
                        "dimensional",
                    ),
                    "origin-rebase": ("precision", "represent", "resolution"),
                }
                assert any(word in str(error).lower() for word in patterns[identity])
                record["expected_source_failure"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "stage": record["stage"],
                }
                assert record["stage"] != "device_query", (
                    "source-invalid domain must reject before numerical query"
                )
            finally:
                if plan is not None:
                    plan.close()
                    record["closed_query_stats"] = dict(plan.stats)
                    assert plan.stats["live_bytes"] == 0
                if broad is not None:
                    broad.close()
                    record["closed_broad_stats"] = broad.stats()
            assert canonical(batch.snapshot()) == canonical(before)
            record["after"] = batch.snapshot()
        record["closed_batch_stats"] = batch.stats()


PREFLIGHT = (
    ("bool-gjk", {"gjk_iterations": True}),
    ("gjk-zero", {"gjk_iterations": 0}),
    ("epa-193", {"epa_iterations": 193}),
    ("vertex-three", {"epa_vertex_capacity": 3}),
    ("face-129", {"epa_face_capacity": 129}),
    ("pair-negative", {"pair_capacity": -1}),
    ("memory-zero", {"memory_limit_bytes": 0}),
    ("duplicate-reverse", {}),
)


@pytest.mark.tyx
@pytest.mark.parametrize(
    "identity,options", PREFLIGHT, ids=[f"K0-API-{name}" for name, _ in PREFLIGHT]
)
def test_k0_api_preflight(identity, options, tyx_mode, tmp_path):
    case = reference().family_case("SS", 0.02)
    with saved(
        tmp_path, f"K0-API-{identity}", tyx_mode, {"case": case, "options": options}
    ) as record:
        with ph.TyxWorldBatch([(17, ordinary_forest())], tyx_mode, cache_root=tmp_path) as batch:
            before = batch.snapshot()
            model, bindings = registrations(case, before[17].topology)
            with batch.prepare_collision(model, bindings) as broad:
                record.update(
                    before=before,
                    model=model,
                    bindings=bindings,
                    stage="public_parameter_preflight",
                )
                keys = tuple(ph.CollisionKey(*key) for key in case["keys"])
                pairs = (
                    (keys, tuple(reversed(keys))) if identity == "duplicate-reverse" else (keys,)
                )
                record["requested_pairs"] = pairs
                with pytest.raises((TypeError, ValueError)) as caught:
                    batch.prepare_convex_query(broad, pairs=pairs, **options)
                record["expected_failure"] = {
                    "type": type(caught.value).__name__,
                    "message": str(caught.value),
                }
                assert broad.last_result is None
                assert canonical(batch.snapshot()) == canonical(before)
                record["after"] = batch.snapshot()
            record["broad_stats"] = broad.stats()
        record["closed_batch_stats"] = batch.stats()
