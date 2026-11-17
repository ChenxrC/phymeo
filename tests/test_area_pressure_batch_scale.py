"""Fixed CUDA batch scales with complete-step correctness and attributable cost.

The public heterogeneous cap is 32 worlds: the preregistered 80-world proposal
is not run, skipped, or emulated by several batches. No timing threshold applies.
The 32-world case is a template prefix, not a constant-ratio scaling experiment.
"""

import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, replace
from time import perf_counter

import numpy as np
import pytest
from test_mujoco_tyx_area_pressure_heterogeneous_scene import H, _declarations
from test_mujoco_tyx_area_pressure_lifecycle import _same

from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
from phymeo.backends.mujoco_heterogeneous_area_scene import (
    MuJoCoTyxHeterogeneousAreaPressureScene,
)
from phymeo.backends.mujoco_resource_budget import MuJoCoTyxResourceBudget
from phymeo.backends.tyx_area_pressure_intersection import TyxAreaPressureIntersector

STEPS = 25
CAPACITY = 64
ROUTES = ("host", "mixed")
EXPANDED_PAYLOAD_LIMIT = 64 * 1024 * 1024


def _optional_array(value):
    return None if value is None else value.tolist()


def _replicas(template, count):
    result = tuple(
        replace(
            template[index % len(template)],
            world=template[index % len(template)].world + 1000 * (index // len(template)),
        )
        for index in range(count)
    )
    assert len({world.world for world in result}) == count
    for index, world in enumerate(result):
        original = template[index % len(template)]
        assert world.model_key == original.model_key
        _same(world.initial_state, original.initial_state)
    return result


def _outcome(world, template_world, replica, is_area):
    return dict(
        world=world.world,
        template_world=template_world,
        replica=replica,
        model_key=world.model_key,
        is_area=is_area,
        accepted_steps=0,
        final_tick=0,
        area_samples_total=0,
        area_empty_steps=0,
        area_nonempty_steps=0,
        area_positive_force_steps=0,
        first_empty_input_tick=None,
        first_nonempty_input_tick=None,
        first_positive_force_input_tick=None,
        routes_on_first_empty=None,
        routes_on_first_positive_force=None,
        peak_area_normal_force_n=0.0,
        final_area_normal_force_n=0.0,
        native_contact_records_total=0,
        peak_native_normal_force_n=0.0,
        final_native_normal_force_n=0.0,
        candidate_route_counts={},
    )


def _record_actual(outcomes, states, query, step):
    routes = {
        int(result.surface.stamp.world_id): tuple(pair_routes)
        for result, pair_routes in zip(query.last_results, query.last_batch_routes, strict=True)
    }
    summary = dict(
        input_tick=step,
        output_tick=step + 1,
        area_empty_worlds=0,
        area_nonempty_worlds=0,
        area_positive_force_worlds=0,
        area_sample_count=0,
        native_contact_count=0,
        total_area_normal_force_n=0.0,
        total_native_normal_force_n=0.0,
    )
    for key, state in states.items():
        outcome = outcomes[key]
        receipt = state.last_step
        assert state.tick == step + 1
        assert receipt is not None and receipt.world == key
        contacts = receipt.contacts
        native_force = float(sum(float(contact.wrench_contact_frame[0]) for contact in contacts))
        assert np.isfinite(native_force)
        outcome["accepted_steps"] += 1
        outcome["final_tick"] = state.tick
        outcome["native_contact_records_total"] += len(contacts)
        outcome["peak_native_normal_force_n"] = max(
            outcome["peak_native_normal_force_n"], native_force
        )
        outcome["final_native_normal_force_n"] = native_force
        summary["native_contact_count"] += len(contacts)
        summary["total_native_normal_force_n"] += native_force
        if not outcome["is_area"]:
            assert key not in routes
            continue
        pairs = receipt.raw_contact_fields["area_pressure"]
        assert len(pairs) == 1  # The unchanged five-world template declares one pair/world.
        pair = pairs[0]
        assert pair["stamp"].world_id == str(key)
        assert pair["stamp"].step_index == step
        assert pair["stamp"].sample_time_s == receipt.input_time
        force = float(np.sum(pair["normal_force"], dtype=np.float64))
        samples = len(pair["sample_ids"])
        assert np.isfinite(force) and force >= 0
        assert pair["normal_force"].shape == (samples,)
        outcome["area_samples_total"] += samples
        outcome["peak_area_normal_force_n"] = max(outcome["peak_area_normal_force_n"], force)
        outcome["final_area_normal_force_n"] = force
        combined_routes = Counter(outcome["candidate_route_counts"])
        combined_routes.update(routes[key])
        outcome["candidate_route_counts"] = dict(combined_routes)
        summary["area_sample_count"] += samples
        summary["total_area_normal_force_n"] += force
        if samples:
            outcome["area_nonempty_steps"] += 1
            summary["area_nonempty_worlds"] += 1
            if outcome["first_nonempty_input_tick"] is None:
                outcome["first_nonempty_input_tick"] = step
        else:
            assert force == 0.0
            outcome["area_empty_steps"] += 1
            summary["area_empty_worlds"] += 1
            if outcome["first_empty_input_tick"] is None:
                outcome["first_empty_input_tick"] = step
                outcome["routes_on_first_empty"] = routes[key]
        if force > 0:
            outcome["area_positive_force_steps"] += 1
            summary["area_positive_force_worlds"] += 1
            if outcome["first_positive_force_input_tick"] is None:
                outcome["first_positive_force_input_tick"] = step
                outcome["routes_on_first_positive_force"] = routes[key]
    return summary


def _timing_summary(timings):
    assert all(len(timings[name]) == STEPS for name in ROUTES)
    return dict(
        first_step_s={name: timings[name][0] for name in ROUTES},
        warm_blocks=[
            dict(
                start_input_tick=start,
                stop_input_tick_exclusive=start + 8,
                steps=8,
                total_s={name: sum(timings[name][start : start + 8]) for name in ROUTES},
            )
            for start in (1, 9, 17)
        ],
        warm_total_s={name: sum(timings[name][1:]) for name in ROUTES},
        all_steps_total_s={name: sum(timings[name]) for name in ROUTES},
    )


@pytest.mark.cuda
@pytest.mark.parametrize("world_count", (5, 20, 32))
def test_fixed_cuda_batch_scale_complete_step_cost_and_host_parity(world_count, tmp_path):
    timings = {name: [] for name in ROUTES}
    payload = dict(
        schema="area-pressure-batch-scale-v1",
        status="RUNNING",
        mode="cuda",
        world_count=world_count,
        steps=STEPS,
        h=H,
        intersector_capacity=CAPACITY,
        fallback="exact_host",
        public_world_limit=32,
        not_run=dict(
            world_count=80, reason="exceeds the current public heterogeneous 32-world limit"
        ),
        replication_rule="repeat template in order; world=template_world+1000*replica; take prefix",
        shared_setup_s={},
        setup_s={},
        complete_step_s=timings,
        execution_orders=[],
        completed_steps=0,
        query_stats_by_step=[],
        accepted_step_summary=[],
        note=(
            "Each timed interval is scene.step + workspace.refresh + readback, including synchronization. "
            "Scientific comparisons and receipt summaries are outside it. Shared declarations, replication, "
            "intersector, batch, scene and workspace initialization are timed separately. No speed gate; "
            "32 worlds has a different model ratio, so no linear-scaling conclusion is implied. "
            "Forces are accepted input-phase receipts, not new geometry solves or inferred forces."
        ),
    )
    query = None
    try:
        started = perf_counter()
        models, template, assets = _declarations()
        payload["shared_setup_s"]["declarations"] = perf_counter() - started
        started = perf_counter()
        worlds = _replicas(template, world_count)
        payload["shared_setup_s"]["world_replication"] = perf_counter() - started
        assert len(template) == 5 and len(models) == 4
        keys = tuple(world.world for world in worlds)
        area_worlds = tuple(world for world in worlds if world.model_key in assets)
        candidates_per_step = sum(
            len(asset.pressure_local.cells) * len(asset.rigid_local.faces)
            for world in area_worlds
            for asset in assets[world.model_key]
        )
        requests_per_step = sum(len(assets[world.model_key]) for world in area_worlds)
        payload["area_world_count"] = len(area_worlds)
        payload["native_world_count"] = world_count - len(area_worlds)
        payload["requests_per_step"] = requests_per_step
        payload["candidate_pairs_per_step"] = candidates_per_step
        payload["model_declarations"] = [
            dict(
                key=model.key,
                contact_capacity=model.contact_capacity,
                constraint_capacity=model.constraint_capacity,
                area_pressure_pairs=model.area_pressure_pairs,
                materials=[asdict(asset.identity) for asset in assets.get(model.key, ())],
            )
            for model in models
        ]
        outcomes = {
            world.world: _outcome(
                world,
                template[index % len(template)].world,
                index // len(template),
                world.model_key in assets,
            )
            for index, world in enumerate(worlds)
        }
        payload["world_outcomes"] = list(outcomes.values())
        with ExitStack() as stack:
            started = perf_counter()
            query = stack.enter_context(
                TyxAreaPressureIntersector(mode="cuda", capacity=CAPACITY, fallback="exact_host")
            )
            payload["shared_setup_s"]["mixed_intersector"] = perf_counter() - started
            batches, scenes, workspaces, budgets = {}, {}, {}, {}
            for name in ROUTES:
                setup = payload["setup_s"][name] = {}
                total_start = perf_counter()
                started = perf_counter()
                budget = (
                    MuJoCoTyxResourceBudget(EXPANDED_PAYLOAD_LIMIT) if world_count > 16 else None
                )
                budgets[name] = budget
                batch = stack.enter_context(
                    MuJoCoTyxHeterogeneousBatch(models, worlds, mode="cuda", resource_budget=budget)
                )
                setup["batch"] = perf_counter() - started
                batches[name] = batch
                started = perf_counter()
                scenes[name] = MuJoCoTyxHeterogeneousAreaPressureScene(
                    batch, assets, intersector=query if name == "mixed" else None
                )
                setup["scene"] = perf_counter() - started
                started = perf_counter()
                workspaces[name] = stack.enter_context(
                    batch.prepare_device_workspace(include_area_pressure=True)
                )
                setup["workspace"] = perf_counter() - started
                setup["total"] = perf_counter() - total_start
                assert batch.world_keys == keys
                assert len({row.bucket for row in batch.layout.values()}) == 3
            payload["bucket_count"] = 3
            payload["world_model_mapping"] = [
                dict(
                    asdict(batches["mixed"].layout[world.world]),
                    template_world=template[index % len(template)].world,
                    replica=index // len(template),
                    initial_q=_optional_array(world.initial_state.q),
                    initial_v=_optional_array(world.initial_state.v),
                    initial_act=_optional_array(world.initial_state.act),
                    initial_time=world.initial_state.time,
                )
                for index, world in enumerate(worlds)
            ]
            outputs, frames = {}, {}
            for step in range(STEPS):
                order = ROUTES if step % 2 == 0 else tuple(reversed(ROUTES))
                payload["execution_orders"].append(order)
                for name in order:
                    batch = batches[name]
                    started = perf_counter()
                    try:
                        accepted = scenes[name].step(H, expected_revision=batch.checkpoint_revision)
                        frames[name] = workspaces[name].refresh(
                            expected_revision=batch.checkpoint_revision
                        )
                        outputs[name] = workspaces[name].readback(frames[name])
                    finally:
                        timings[name].append(perf_counter() - started)
                    assert accepted == dict.fromkeys(keys, True)
                mixed_state = batches["mixed"].integration_snapshot()
                _same(mixed_state, batches["host"].integration_snapshot())
                _same(scenes["mixed"].last_geometry, scenes["host"].last_geometry)
                _same(outputs["mixed"], outputs["host"])
                _same(frames["mixed"].area_catalogs, frames["host"].area_catalogs)
                assert frames["mixed"].ticks == frames["host"].ticks == (step + 1,) * world_count
                payload["accepted_step_summary"].append(
                    _record_actual(outcomes, mixed_state, query, step)
                )
                payload["query_stats_by_step"].append(dict(query.stats))
                payload["completed_steps"] = step + 1
            stats = dict(query.stats)
            payload["query_stats_after_steps"] = stats
            assert stats["batch_calls"] == STEPS
            assert stats["attempts"] == STEPS * requests_per_step
            assert stats["candidate_pairs"] == STEPS * candidates_per_step
            assert (
                stats["device_polygons"] + stats["fallback_pairs"] + stats["culled_pairs"]
                == stats["candidate_pairs"]
            )
            assert stats["device_polygons"] > 0 and stats["culled_pairs"] > 0
            assert (
                0 < stats["launches"] <= STEPS * ((candidates_per_step + CAPACITY - 1) // CAPACITY)
            )
            for outcome in outcomes.values():
                assert outcome["accepted_steps"] == outcome["final_tick"] == STEPS
                if outcome["is_area"]:
                    assert outcome["area_empty_steps"] + outcome["area_nonempty_steps"] == STEPS
                if outcome["template_world"] == 305:
                    assert outcome["first_empty_input_tick"] == 0
                    assert outcome["first_nonempty_input_tick"] is not None
                    assert outcome["first_positive_force_input_tick"] is not None
                    assert (
                        0
                        < outcome["first_nonempty_input_tick"]
                        <= outcome["first_positive_force_input_tick"]
                        < STEPS
                    )
            assert any(row["peak_area_normal_force_n"] > 0 for row in outcomes.values())
            assert any(
                row["native_contact_records_total"] > 0
                for row in outcomes.values()
                if not row["is_area"]
            )
            payload["timing_summary"] = _timing_summary(timings)
            payload["registered_cuda_payload"] = {
                name: None
                if budget is None
                else {
                    key: budget.snapshot()[key]
                    for key in (
                        "scope",
                        "device_payload_limit_bytes",
                        "current_bytes",
                        "peak_bytes",
                        "reserved_bytes",
                        "unknown_retained_bytes",
                        "excluded",
                    )
                }
                for name, budget in budgets.items()
            }
        payload["query_stats_after_close"] = dict(query.stats)
        assert query.stats["live_bytes"] == 0
        payload["status"] = "PASS"
    except BaseException as error:
        payload["status"] = "FAIL"
        payload["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        # Preserve same-run partial evidence on a functional failure; do not
        # retry a scale or fabricate unavailable timing/force observations.
        (tmp_path / "cost.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8"
        )
