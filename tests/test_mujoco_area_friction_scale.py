"""One fixed 1/4/32-world live-friction sequence; cost is not a speed claim."""

import gzip
import json
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields, is_dataclass, replace
from time import perf_counter

import numpy as np
import pytest
from test_mujoco_area_friction import _equations, _plain
from test_mujoco_area_friction_live import H, _cases, _published

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxStepReceipt
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_resource_budget import MuJoCoTyxResourceBudget
from phymeo.backends.mujoco_tyx_state import MuJoCoTyxIntegrationState
from phymeo.geometry.area_pressure import AreaPressureStamp

COUNTS = (1, 4, 32)
STEPS = 3
DEVICE_PAYLOAD_LIMIT = 64 * 1024 * 1024  # Same explicit policy for all three sizes.


def _same_physics(actual, expected):
    """No raw values removed: only explicitly checked world/owner identities vary."""
    if isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            if type(expected) is AreaPressureStamp and field.name in (
                "world_id",
                "configuration_id",
            ):
                continue
            if (
                type(expected) in (MuJoCoTyxStepReceipt, MuJoCoTyxIntegrationState)
                and field.name == "world"
            ):
                continue
            _same_physics(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping) and set(actual) == set(expected)
        for key in expected:
            _same_physics(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            _same_physics(a, b)
    else:
        assert type(actual) is type(expected) and actual == expected


def _timing(rows, count):
    complete = [row["complete_observed_step_s"] for row in rows]
    assert len(rows) == STEPS and all(t > 0 and np.isfinite(t) for t in complete)
    return dict(
        first_step=rows[0],
        next_two_steps=rows[1:],
        next_two_total_s=sum(complete[1:]),
        all_three_total_s=sum(complete),
        next_two_worldsteps_per_s=2 * count / sum(complete[1:]),
        all_three_worldsteps_per_s=3 * count / sum(complete),
        throughput_scope="scene.step + refresh + readback + integration_snapshot; excludes independent audit/save",
    )


@pytest.mark.cuda
def test_fixed_one_four_thirtytwo_world_full_output_scaling(tmp_path):
    payload = dict(
        schema="mujoco-area-friction-scale-v1",
        mode="cuda",
        h=H,
        world_counts=COUNTS,
        steps_per_case=STEPS,
        cases=[],
        cache_policy="existing process/JIT caches retained; first measured step is not a cold-cache claim",
        comparison="same model, initial state, law, finite surface and complete numeric output; W1 same-run oracle",
        timing_gate=None,
        external_solver_comparison=None,
    )
    reference = []
    try:
        started = perf_counter()
        all_specs, template, all_assets = _cases()
        payload["shared_fixture_declarations_s"] = perf_counter() - started
        spec, initial = all_specs[0], template[0]
        assets = {spec.key: all_assets[spec.key]}
        payload.update(
            model=dict(
                key=spec.key,
                source=spec.compilation.source,
                contact_capacity=spec.contact_capacity,
                constraint_capacity=spec.constraint_capacity,
                area_pressure_pairs=spec.area_pressure_pairs,
                area_friction_laws=spec.area_friction_laws,
            ),
            assets=assets,
            initial_state=initial.initial_state,
            declarations_note="Original B1 fixture builds two native sources; only its FREE mu=.5 model is allocated in every timed case",
        )
        for count in COUNTS:
            case = dict(
                world_count=count,
                records=[],
                timings=[],
                setup={},
                completed_steps=0,
                accepted_steps=0,
            )
            payload["cases"].append(case)
            started = perf_counter()
            worlds = tuple(replace(initial, world=7 + 1000 * i) for i in range(count))
            case["setup"]["world_replication_s"] = perf_counter() - started
            case["worlds"] = worlds
            keys = tuple(world.world for world in worlds)
            with ExitStack() as stack:
                budget = MuJoCoTyxResourceBudget(DEVICE_PAYLOAD_LIMIT)
                setup_start = perf_counter()
                started = perf_counter()
                batch = stack.enter_context(
                    MuJoCoTyxHeterogeneousBatch(
                        (spec,), worlds, mode="cuda", resource_budget=budget
                    )
                )
                case["setup"]["batch_s"] = perf_counter() - started
                started = perf_counter()
                scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
                case["setup"]["scene_s"] = perf_counter() - started
                started = perf_counter()
                ws = stack.enter_context(
                    batch.prepare_device_workspace(include_bodies=True, include_area_pressure=True)
                )
                case["setup"]["workspace_s"] = perf_counter() - started
                case["setup"]["owned_total_s"] = perf_counter() - setup_start
                assert (
                    batch.world_keys == keys
                    and len({row.bucket for row in batch.layout.values()}) == 1
                )
                case.update(
                    layout=batch.layout,
                    initial=batch.integration_snapshot(),
                    observation_binding=ws.binding,
                    model_metadata=ws.model_metadata,
                    resources_before=batch.resource_snapshot(),
                    workspace_before=ws.counters,
                    counters_before=batch.counters,
                )
                for step in range(STEPS):
                    configs = {key: batch.area_pressure_configuration_id(key) for key in keys}
                    interval = case["timings"]
                    record = dict(
                        step=step, state_accepted=False, observation_complete=False, certificates={}
                    )
                    case["records"].append(record)
                    started = perf_counter()
                    accepted = scene.step(H, expected_revision=batch.checkpoint_revision)
                    physics_done = perf_counter()
                    case["accepted_steps"] = step + 1
                    record.update(
                        state_accepted=True,
                        accepted=accepted,
                        receipt=batch.last_step,
                        states=batch.last_step.states,
                        geometry=scene.last_geometry,
                    )
                    frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                    refresh_done = perf_counter()
                    arrays = ws.readback(frame)
                    copy_done = perf_counter()
                    states = batch.integration_snapshot()
                    finished = perf_counter()
                    interval.append(
                        dict(
                            scene_step_s=physics_done - started,
                            observation_refresh_s=refresh_done - physics_done,
                            observation_readback_s=copy_done - refresh_done,
                            integration_snapshot_s=finished - copy_done,
                            complete_observed_step_s=finished - started,
                        )
                    )
                    record.update(
                        frame=frame, arrays=arrays, states=states, observation_complete=True
                    )
                    assert accepted == dict.fromkeys(keys, True)
                    assert frame.ticks == (step + 1,) * count
                    assert set(states) == set(keys) and set(scene.last_geometry) == set(keys)
                    for row_index, key in enumerate(keys):
                        state = states[key]
                        assert state.world == state.last_step.world == key
                        assert state.tick == step + 1 and state.last_step.tick == step + 1
                        rows, solver, problem = _published(state.last_step)
                        for pair, geometry, catalog in zip(
                            rows, scene.last_geometry[key], frame.area_catalogs[key], strict=True
                        ):
                            stamp = pair["stamp"]
                            assert (
                                stamp.world_id == str(key)
                                and stamp.configuration_id == configs[key]
                            )
                            assert (
                                stamp.step_index == step
                                and stamp.sample_time_s == state.last_step.input_time
                            )
                            assert geometry.surface.stamp == stamp == catalog["stamp"]
                            assert (
                                pair["identity"]
                                == spec.area_friction_laws[0].identity
                                == catalog["identity"]
                            )
                        cert = _equations(problem, state.v)
                        assert cert["residual_norm"] <= 2e-5 * cert["scale"]
                        record["certificates"][key] = dict(
                            residual_norm=cert["residual_norm"],
                            scale=cert["scale"],
                            strict_host_gate=cert["residual_norm"] <= 1e-10 * cert["scale"],
                        )
                        assert sum(len(pair["sample_ids"]) for pair in rows) > 0
                        assert sum(float(np.sum(pair["normal_force"])) for pair in rows) > 0
                        values = {name: value[row_index].copy() for name, value in arrays.items()}
                        candidate = dict(
                            state=state,
                            geometry=scene.last_geometry[key],
                            catalog=frame.area_catalogs[key],
                            arrays=values,
                        )
                        if count == 1:
                            reference.append(candidate)
                        else:
                            _same_physics(candidate, reference[step])
                        assert state.q.tobytes() != case["initial"][key].q.tobytes()
                    case["completed_steps"] = step + 1
                case.update(
                    timing_summary=_timing(case["timings"], count),
                    counters_after=batch.counters,
                    workspace_after=ws.counters,
                    resources_after=batch.resource_snapshot(),
                    final=batch.integration_snapshot(),
                )
                case["payload_bytes"] = dict(
                    solver_manifest=case["resources_after"]["manifest_payload_bytes"],
                    observation_workspace=ws.counters["allocated_payload_bytes"],
                    exclusion="HOST arrays, native models, CUDA context/JIT and allocator overhead are not included",
                )
                assert (
                    case["resources_before"]["manifest_payload_bytes"]
                    == case["resources_after"]["manifest_payload_bytes"]
                )
                assert (
                    case["workspace_before"]["allocated_payload_bytes"]
                    == case["workspace_after"]["allocated_payload_bytes"]
                )
                assert case["counters_after"]["completed_steps"] == STEPS
                assert case["counters_after"]["accepted_world_steps"] == STEPS * count
                case["status"] = "PASS"
            case["closed_resources"] = batch.resource_snapshot()
            case["closed_budget"] = budget.snapshot()
            assert (
                case["closed_budget"]["current_bytes"]
                == case["closed_budget"]["reserved_bytes"]
                == 0
            )
            assert case["closed_resources"]["closed"] and not case["closed_resources"]["buckets"]
        payload["status"] = "PASS"
    except BaseException as error:
        payload.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        with gzip.open(tmp_path / "scale-full.json.gz", "xt", encoding="utf-8") as stream:
            json.dump(_plain(payload), stream, allow_nan=False)
        summary = dict(
            schema=payload["schema"],
            status=payload.get("status"),
            cache_policy=payload["cache_policy"],
            shared_fixture_declarations_s=payload.get("shared_fixture_declarations_s"),
            cases=[
                {
                    name: case[name]
                    for name in (
                        "world_count",
                        "accepted_steps",
                        "completed_steps",
                        "setup",
                        "timings",
                        "timing_summary",
                        "payload_bytes",
                        "status",
                    )
                    if name in case
                }
                for case in payload["cases"]
            ],
        )
        (tmp_path / "cost.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
