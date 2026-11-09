"""K0 Q08 frozen resident query protocols (17 identities, W32 deferred).

AUTHORING ONLY / UNEXECUTED. Two warm queries are not a physical window.
All three fault hooks call the actual stage before corrupting/raising.
"""

import importlib.util
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph

H = 1 / 1024
PROTOCOLS = tuple(
    f"w{worlds}-{density}" for worlds in (1, 4, 32) for density in ("sparse", "dense")
) + (
    "filters-samebody",
    "filters-mask",
    "filters-excluded",
    "epoch_bank",
    "pair_capacity",
    "budget_one",
    "fault-raw_nan",
    "fault-raw_bad_id",
    "fault-late_record",
    "owner_close_rebuild",
    "construction_failure",
)


@lru_cache(maxsize=1)
def common():
    path = Path(__file__).resolve().parents[1] / "collision" / "test_device_convex_query.py"
    spec = importlib.util.spec_from_file_location("k0_frozen_query_test_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forest_spec(count, density="dense", *, moving=False):
    q = (0.0, 0.09, 0.23) if density == "dense" else (0.0, 0.5, 1.0)
    worlds = tuple(17 + 37 * index for index in range(count))
    velocity = (1 / 32, 0.0, 0.0) if moving else (0.0, 0.0, 0.0)
    return (
        worlds,
        q,
        [(world, common().ordinary_forest(3, q=q, velocity=velocity)) for world in worlds],
    )


def scene(checkpoint, worlds, *, filter_kind=None):
    shapes, bindings, exclusions = [], [], []
    registry = {}
    for world in worlds:
        for group, (kind, body, number, first_id) in enumerate(
            zip(("S", "C", "H"), (11, 29, 71), (3, 5, 7), (100, 200, 300), strict=True)
        ):
            points, radius, _ = common().reference().primitive(kind)
            collision_group, mask = 1, 2**31 - 1
            actual_group = group
            if world == 54 and filter_kind == "samebody" and kind == "C":
                body, actual_group = 11, 0
                points = points + (0, 0, 0.09)
            if world == 54 and filter_kind == "mask":
                collision_group = mask = (1, 2, 3)[group]
            key = ph.CollisionKey(world, body, number)
            shape = ph.CollisionShape(
                key,
                {"S": "sphere", "C": "capsule", "H": "hull"}[kind],
                tuple(range(first_id, first_id + len(points))),
                radius,
                group=collision_group,
                mask=mask,
                revision=7 + group,
            )
            shapes.append(shape)
            bindings.append(
                ph.ResidentTreeBinding(
                    key, world, actual_group, points, topology=checkpoint[world].topology
                )
            )
            registry[(world, kind)] = (shape, points, actual_group)
        if world == 54 and filter_kind == "excluded":
            exclusions.append((registry[(world, "S")][0].key, registry[(world, "C")][0].key))
    return ph.CollisionModel(tuple(shapes), tuple(exclusions)), tuple(bindings), registry


def pair_case(first, second, q, registry):
    entries = {entry[0].key: (kind, entry) for (_, kind), entry in registry.items()}
    kinds, local, positions, radii, ids, points = [], [], [], [], [], []
    for key in (first, second):
        kind, (shape, original, group) = entries[key]
        position = np.array([0.0, 0.0, q[group]])
        kinds.append(kind)
        local.append(original)
        positions.append(position)
        points.append(original + position)
        radii.append(shape.radius)
        ids.append(shape.vertex_ids)
    normal = np.array([0.0, 0.0, 1.0])
    gap = float(min(points[1] @ normal) - radii[1] - max(points[0] @ normal) - radii[0])
    return {
        "family": "".join(kinds),
        "kinds": tuple(kinds),
        "keys": tuple((key.world, key.body, key.shape) for key in (first, second)),
        "ids": tuple(ids),
        "local": tuple(local),
        "positions": tuple(positions),
        "rotations": (np.eye(3), np.eye(3)),
        "radii": tuple(radii),
        "points": tuple(points),
        "gap_m": gap,
        "normal": normal,
        "requested_reverse": False,
        "registration_reverse": False,
        "oracle": "independent forest q and finite parallel support planes",
    }


def expected_pairs(worlds, registry, density="dense", filter_kind=None):
    result = []
    if density == "dense":
        for world in worlds:
            if world != 54 or filter_kind is None:
                result.append(
                    tuple(sorted((registry[(world, "S")][0].key, registry[(world, "C")][0].key)))
                )
            result.append(
                tuple(sorted((registry[(world, "C")][0].key, registry[(world, "H")][0].key)))
            )
    return tuple(sorted(result))


def allowed_pairs(worlds, registry, filter_kind=None):
    pairs = []
    for world in worlds:
        for a, b in (("S", "C"), ("S", "H"), ("C", "H")):
            if world == 54 and filter_kind is not None and (a, b) == ("S", "C"):
                continue
            pairs.append(tuple(sorted((registry[(world, a)][0].key, registry[(world, b)][0].key))))
    return tuple(sorted(pairs))


def check_forest_result(
    result, model, bindings, before, worlds, q, registry, *, density="dense", filter_kind=None
):
    c = common()
    c.check_raw(result)
    c.check_source(result, model, bindings, before)
    expected = expected_pairs(worlds, registry, density, filter_kind)
    assert tuple(row.pair for row in result.rows) == expected
    assert result.pair_source == "host_broadphase"
    assert tuple(result.requested_pairs) == expected
    assert tuple(result.request_row) == tuple(range(len(expected)))
    assert tuple(result.request_reversed) == (False,) * len(expected)
    assert tuple(result.source["allowed_pairs"]) == allowed_pairs(worlds, registry, filter_kind)
    assert tuple(result.world_keys) == worlds
    checked = []
    for row in result.rows:
        case = pair_case(*row.pair, q, registry)
        checked.append(c.check_case_row(row, case, result.source))
    return checked


def attempt(batch, plan, record, label, *, current=None):
    if current is None:
        return common().query_attempt(batch, plan, record, label)
    item = {
        "label": label,
        "kind": "require_current",
        "returned": False,
        "error": None,
        "input_result": current,
        "stats_before": dict(plan.stats),
    }
    record["attempts"].append(item)
    try:
        result = plan.require_current(current)
        item["returned"] = True
        return result
    except BaseException as error:
        item["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        item["stats_after"] = dict(plan.stats)


@contextmanager
def prepared(tmp_path, mode, count=1, density="dense", *, filter_kind=None, moving=False):
    worlds, q, specs = forest_spec(count, density, moving=moving)
    with ph.TyxWorldBatch(specs, mode, cache_root=tmp_path) as batch:
        before = batch.snapshot()
        model, bindings, registry = scene(before, worlds, filter_kind=filter_kind)
        with batch.prepare_collision(
            model, bindings, candidate_capacity=3 * count, tree_dynamic_pairs=True
        ) as broad:
            yield batch, broad, before, model, bindings, registry, worlds, q


def geometry_result(result):
    return common().canonical(
        {
            "rows": result.rows,
            "raw_geometry": {key: value for key, value in result.raw.items() if key != "meta"},
            "source": result.source,
            "ticks": result.ticks,
            "shape_keys": result.shape_keys,
        }
    )


def repeated_queries(identity, mode, tmp_path, record):
    count = int(identity.split("-")[0][1:])
    density = identity.split("-")[1]
    with prepared(tmp_path, mode, count, density) as (
        batch,
        broad,
        before,
        model,
        bindings,
        registry,
        worlds,
        q,
    ):
        record.update(
            before=before, model=model, bindings=bindings, declared_q=q, world_keys=worlds
        )
        with closing(batch.prepare_convex_query(broad)) as plan:
            stats_before = dict(plan.stats)
            first = attempt(batch, plan, record, "query_1")
            record["checks_1"] = check_forest_result(
                first, model, bindings, before, worlds, q, registry, density=density
            )
            owned = common().canonical(first)
            warm_before = dict(plan.stats)
            second = attempt(batch, plan, record, "query_2")
            warm_after = dict(plan.stats)
            record["checks_2"] = check_forest_result(
                second, model, bindings, before, worlds, q, registry, density=density
            )
            assert common().canonical(first) == owned
            assert geometry_result(first) == geometry_result(second)
            assert second.state_epoch == first.state_epoch and second.active == first.active
            assert second.attempt_tag > first.attempt_tag
            assert warm_after["allocations"] == warm_before["allocations"]
            assert warm_after["static_uploads"] == warm_before["static_uploads"]
            assert warm_after["queries"] - stats_before["queries"] == 2
            assert broad.last_result.candidates == expected_pairs(worlds, registry, density)
            assert broad.stats()["queries"] == 2
            record.update(
                warm_before=warm_before,
                warm_after=warm_after,
                after=batch.snapshot(),
                broad_last=broad.last_result,
                query_stats=dict(plan.stats),
                batch_stats=batch.stats(),
            )
            assert common().canonical(record["after"]) == common().canonical(before)
            if count == 32:
                assert (
                    len(model.shapes) == 96
                    and sum(len(shape.vertex_ids) for shape in model.shapes) == 416
                )
        record["closed_query_stats"] = dict(plan.stats)
        assert plan.stats["live_bytes"] == 0
    record["closed_batch_stats"] = batch.stats()


def filter_query(identity, mode, tmp_path, record):
    filter_kind = identity.split("-", 1)[1]
    with prepared(tmp_path, mode, 4, filter_kind=filter_kind) as (
        batch,
        broad,
        before,
        model,
        bindings,
        registry,
        worlds,
        q,
    ):
        record.update(
            before=before, model=model, bindings=bindings, declared_q=q, filter_kind=filter_kind
        )
        with closing(batch.prepare_convex_query(broad)) as plan:
            result = attempt(batch, plan, record, "filtered_query")
            record["checks"] = check_forest_result(
                result, model, bindings, before, worlds, q, registry, filter_kind=filter_kind
            )
            assert len(result.rows) == 7
            assert common().canonical(batch.snapshot()) == common().canonical(before)
        record["closed_query_stats"] = dict(plan.stats)
    record["closed_batch_stats"] = batch.stats()


def epoch_query(mode, tmp_path, record):
    with prepared(tmp_path, mode, moving=True) as (
        batch,
        broad,
        before,
        model,
        bindings,
        registry,
        worlds,
        q,
    ):
        record.update(
            before=before,
            model=model,
            bindings=bindings,
            declared_q=q,
            h_s=H,
            initial_velocity=(1 / 32, 0, 0),
        )
        with closing(batch.prepare_convex_query(broad)) as plan:
            first = attempt(batch, plan, record, "initial_query")
            saved_first = common().canonical(first)
            record["checks_initial"] = check_forest_result(
                first, model, bindings, before, worlds, q, registry
            )
            accepted = batch.step(H)
            assert accepted == {17: True}
            after = batch.snapshot()
            assert after[17].tick == 1 and abs(after[17].position[0] - 1 / 32768) <= 1e-12
            np.testing.assert_allclose(after[17].velocity, (1 / 32, 0, 0), atol=1e-12, rtol=0)
            with pytest.raises(ValueError, match="stale|epoch"):
                attempt(batch, plan, record, "old_result_after_step", current=first)
            second = attempt(batch, plan, record, "after_step_query")
            assert second.active != first.active and second.state_epoch > first.state_epoch
            advanced_q = (1 / 32768, q[1], q[2])
            record["checks_advanced"] = check_forest_result(
                second, model, bindings, after, worlds, advanced_q, registry
            )
            batch.restore(before)
            with pytest.raises(ValueError, match="stale|epoch"):
                attempt(batch, plan, record, "old_result_after_restore", current=first)
            third = attempt(batch, plan, record, "restored_query")
            assert third.state_epoch > second.state_epoch and tuple(third.ticks) == (0,)
            assert common().canonical(batch.snapshot()) == common().canonical(before)
            assert geometry_result(first) == geometry_result(third)
            assert common().canonical(first) == saved_first
            record.update(
                accepted=accepted,
                after_step=after,
                restored=batch.snapshot(),
                checks_restored=check_forest_result(
                    third, model, bindings, before, worlds, q, registry
                ),
            )
        record["closed_query_stats"] = dict(plan.stats)
    record["closed_batch_stats"] = batch.stats()


def capacity_query(mode, tmp_path, record):
    with prepared(tmp_path, mode) as (batch, broad, before, model, bindings, registry, worlds, q):
        record.update(before=before, model=model, bindings=bindings, capacities=(2, 1, 2))
        with closing(batch.prepare_convex_query(broad, pair_capacity=2)) as first_plan:
            first = attempt(batch, first_plan, record, "capacity_2")
            check_forest_result(first, model, bindings, before, worlds, q, registry)
            owned = common().canonical(first)
            broad_before = broad.last_result
            with closing(batch.prepare_convex_query(broad, pair_capacity=1)) as limited:
                with pytest.raises(ph.ConvexQueryCapacityExceeded) as caught:
                    attempt(batch, limited, record, "capacity_1")
                assert caught.value.required == 2 and caught.value.capacity == 1
                assert dict(caught.value.per_world_required) == {17: 2}
                assert limited.last_result is None and broad.last_result is broad_before
                record["overflow"] = {
                    "required": caught.value.required,
                    "capacity": caught.value.capacity,
                    "per_world_required": dict(caught.value.per_world_required),
                }
            with closing(batch.prepare_convex_query(broad, pair_capacity=2)) as retry_plan:
                retry = attempt(batch, retry_plan, record, "fresh_capacity_2_retry")
                check_forest_result(retry, model, bindings, before, worlds, q, registry)
                assert geometry_result(first) == geometry_result(retry)
            assert common().canonical(first) == owned
            assert common().canonical(batch.snapshot()) == common().canonical(before)
        record["closed_query_stats"] = [
            dict(plan.stats) for plan in (first_plan, limited, retry_plan)
        ]
        assert all(stats["live_bytes"] == 0 for stats in record["closed_query_stats"])
    record["closed_batch_stats"] = batch.stats()


def budget_query(mode, tmp_path, record):
    c = common()
    case = c.reference().transformed(c.reference().family_case("HH", -0.01), "tilt")
    record["declared_case"] = case
    with ph.TyxWorldBatch(
        [(17, c.ordinary_forest(anchor=case["positions"][0], orientation=c.case_quaternion(case)))],
        mode,
        cache_root=tmp_path,
    ) as batch:
        before = batch.snapshot()
        model, bindings = c.registrations(case, before[17].topology)
        record.update(before=before, model=model, bindings=bindings)
        with batch.prepare_collision(model, bindings) as broad:
            pairs = (tuple(ph.CollisionKey(*key) for key in case["keys"]),)
            with closing(
                batch.prepare_convex_query(broad, pairs=pairs, gjk_iterations=1, epa_iterations=1)
            ) as plan:
                result = attempt(batch, plan, record, "budget_one")
                c.check_raw(result)
                c.check_source(result, model, bindings, before, case=case)
                assert len(result.rows) == 1
                row = result.rows[0]
                assert row.gjk_used <= 1 and row.epa_used <= 1
                assert row.relation != "SEPARATED"
                if row.status == "RESOLVED":
                    record["checks"] = c.check_case_row(row, case, result.source)
                else:
                    assert row.status in ("UNKNOWN", "RESOURCE_LIMIT")
                    assert row.reason in (
                        "GJK_LIMIT",
                        "EPA_LIMIT",
                        "EPA_CAPACITY",
                        "DEGENERATE_SIMPLEX",
                        "STALLED",
                    )
                assert c.canonical(batch.snapshot()) == c.canonical(before)
            record["closed_query_stats"] = dict(plan.stats)
    record["closed_batch_stats"] = batch.stats()


def fault_query(identity, mode, tmp_path, record, monkeypatch):
    fault = identity.split("-", 1)[1]
    c = common()
    with prepared(tmp_path, mode) as (batch, broad, before, model, bindings, registry, worlds, q):
        record.update(before=before, model=model, bindings=bindings, fault=fault)
        with closing(batch.prepare_convex_query(broad)) as plan:
            first = attempt(batch, plan, record, "before_fault")
            check_forest_result(first, model, bindings, before, worlds, q, registry)
            owned = c.canonical(first)
            broad_before = broad.last_result
            if fault == "late_record":
                original = plan._build_result

                def injected(*args, **kwargs):
                    value = original(*args, **kwargs)
                    record["late_owned_before_failure"] = value
                    raise MemoryError("K0 frozen late owned result construction failure")

                hook = "_build_result"
                expected_error = MemoryError
            else:
                original = plan._read_raw

                def injected(*args, **kwargs):
                    raw = original(*args, **kwargs)
                    record["raw_before_injection"] = c.encode(raw)
                    altered = {key: np.array(value, copy=True) for key, value in raw.items()}
                    if fault == "raw_nan":
                        altered["gap"][0, 0] = np.nan
                    else:
                        altered["material_ids"][0, 0, 0] = 2**31 - 1
                    record["raw_after_injection"] = c.encode(altered)
                    return altered

                hook = "_read_raw"
                expected_error = (ValueError, RuntimeError)
            with monkeypatch.context() as patch:
                patch.setattr(plan, hook, injected)
                with pytest.raises(expected_error):
                    attempt(batch, plan, record, "actual_raw_or_owned_fault")
            assert plan.last_result is first and broad.last_result is broad_before
            assert c.canonical(first) == owned and c.canonical(batch.snapshot()) == c.canonical(
                before
            )
            assert plan.require_current(first) is first
            retry = attempt(batch, plan, record, "same_input_retry")
            record["retry_checks"] = check_forest_result(
                retry, model, bindings, before, worlds, q, registry
            )
            assert geometry_result(retry) == geometry_result(first)
            assert retry.attempt_tag >= first.attempt_tag + 2
            assert plan.stats["failed_queries"] == 1
            record.update(after=batch.snapshot(), failed_costs=dict(plan.stats))
        record["closed_query_stats"] = dict(plan.stats)
    record["closed_batch_stats"] = batch.stats()


def owner_close(mode, tmp_path, record):
    c = common()
    with prepared(tmp_path / "first", mode) as (
        batch,
        broad,
        before,
        model,
        bindings,
        registry,
        worlds,
        q,
    ):
        with prepared(tmp_path / "second", mode) as (
            other,
            other_broad,
            other_before,
            _,
            _,
            _,
            _,
            _,
        ):
            record.update(before=before, other_before=other_before, model=model, bindings=bindings)
            plan = batch.prepare_convex_query(broad)
            other_plan = other.prepare_convex_query(other_broad)
            fresh = None
            try:
                first = attempt(batch, plan, record, "first_owner_query")
                check_forest_result(first, model, bindings, before, worlds, q, registry)
                owned = c.canonical(first)
                with pytest.raises((ValueError, TypeError), match="owner|batch"):
                    attempt(other, plan, record, "wrong_owner")
                assert other_plan.last_result is None and other_broad.last_result is None
                plan.close()
                with pytest.raises((ValueError, RuntimeError), match="clos"):
                    attempt(batch, plan, record, "closed_plan")
                assert plan.stats["live_bytes"] == 0
                fresh = batch.prepare_convex_query(broad)
                result = attempt(batch, fresh, record, "fresh_plan_same_inputs")
                check_forest_result(result, model, bindings, before, worlds, q, registry)
                assert geometry_result(first) == geometry_result(result)
                assert c.canonical(first) == owned
                assert c.canonical(batch.snapshot()) == c.canonical(before)
                assert c.canonical(other.snapshot()) == c.canonical(other_before)
                broad.close()
                assert fresh.stats["live_bytes"] == 0
                other.close()
                assert other_plan.stats["live_bytes"] == 0
            finally:
                plan.close()
                other_plan.close()
                if fresh is not None:
                    fresh.close()
                record["closed_query_stats"] = [
                    dict(item.stats) for item in (plan, other_plan, fresh) if item is not None
                ]
        record["other_closed_batch_stats"] = other.stats()
    record["closed_batch_stats"] = batch.stats()


def construction_failure(mode, tmp_path, record, monkeypatch):
    from phymeo.backends._device_convex_query import ConvexQueryPlan

    c = common()
    with prepared(tmp_path, mode) as (batch, broad, before, model, bindings, _, _, _):
        record.update(before=before, model=model, bindings=bindings)
        original = ConvexQueryPlan._allocate
        captured = []

        def allocate(self, *args, **kwargs):
            value = original(self, *args, **kwargs)
            captured.append(self)
            record.setdefault("completed_allocations", []).append(
                {"args": args, "kwargs": kwargs, "stats": dict(self.stats)}
            )
            if len(captured) == 2:
                raise MemoryError("K0 frozen post-registration allocation interruption")
            return value

        with monkeypatch.context() as patch:
            patch.setattr(ConvexQueryPlan, "_allocate", allocate)
            with pytest.raises(MemoryError, match="post-registration") as caught:
                batch.prepare_convex_query(broad)
        assert len(captured) == 2 and captured[0] is captured[1]
        assert captured[0].stats["live_bytes"] == 0
        assert captured[0].stats["allocations"] >= 2
        assert broad.last_result is None and c.canonical(batch.snapshot()) == c.canonical(before)
        record.update(
            expected_failure=str(caught.value),
            failed_plan_stats=dict(captured[0].stats),
            after=batch.snapshot(),
        )
    record["closed_batch_stats"] = batch.stats()


@pytest.mark.tyx
@pytest.mark.parametrize("identity", PROTOCOLS, ids=lambda identity: f"K0-Q08-{identity}")
def test_resident_convex_protocol(identity, tyx_mode, tmp_path, monkeypatch):
    inputs = {
        "identity": identity,
        "world_formula": "17+37*j",
        "parents": (-1, -1, -1),
        "axis": (0, 0, 1),
        "mass_kg": (1, 1, 1),
        "inertia_kg_m2": (1 / 16,) * 3,
        "gravity": (0, 0, 0),
        "drive": (0, 0, 0),
        "default_q_dense": (0, 0.09, 0.23),
        "default_q_sparse": (0, 0.5, 1),
        "h_s": H,
    }
    with common().saved(tmp_path, f"K0-Q08-{identity}", tyx_mode, inputs) as record:
        path = Path(__file__).resolve()
        record["source"].append(
            {
                "path": str(path),
                "sha256": common().hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes_hex": path.read_bytes().hex(),
            }
        )
        record["stage"] = "frozen_protocol"
        if identity.startswith("w"):
            repeated_queries(identity, tyx_mode, tmp_path, record)
        elif identity.startswith("filters-"):
            filter_query(identity, tyx_mode, tmp_path, record)
        elif identity == "epoch_bank":
            epoch_query(tyx_mode, tmp_path, record)
        elif identity == "pair_capacity":
            capacity_query(tyx_mode, tmp_path, record)
        elif identity == "budget_one":
            budget_query(tyx_mode, tmp_path, record)
        elif identity.startswith("fault-"):
            fault_query(identity, tyx_mode, tmp_path, record, monkeypatch)
        elif identity == "owner_close_rebuild":
            owner_close(tyx_mode, tmp_path, record)
        else:
            assert identity == "construction_failure"
            construction_failure(tyx_mode, tmp_path, record, monkeypatch)
        expected_attempts = (
            2
            if identity.startswith("w")
            else 1
            if identity.startswith("filters-") or identity == "budget_one"
            else 5
            if identity == "epoch_bank"
            else 3
            if identity == "pair_capacity" or identity.startswith("fault-")
            else 4
            if identity == "owner_close_rebuild"
            else 0
        )
        assert len(record["attempts"]) == expected_attempts
        record["declared_attempt_count"] = expected_attempts
