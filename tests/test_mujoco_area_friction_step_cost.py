"""Measure ordinary and phase-timed CUDA steps in counterbalanced order."""

import json
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from time import perf_counter_ns

import numpy as np
import pytest
from test_area_pressure_scene_preparation import _validation_calls
from test_area_pressure_step_mesh_reuse import uncached_inputs
from test_mujoco_area_friction_lifecycle import _same
from test_mujoco_area_friction_live import H, _cases
from test_mujoco_area_friction_scale import DEVICE_PAYLOAD_LIMIT
from test_mujoco_area_friction_wide import _wide

import phymeo.backends.mujoco_area_pressure_scene as scene_geometry
from experiments.area_pressure import moving_jaw_grasp as jaws
from experiments.area_pressure.scene_step_timing import measure_scene_step
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_resource_budget import MuJoCoTyxResourceBudget
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState


def _workload(case):
    if case == "jaws":
        specs, worlds, assets = jaws._declarations()
        phases = [
            (name, count, MuJoCoControl((drive, drive), qfrc_applied=(0, 0, force, 0, 0, 0, 0, 0)))
            for name, count, drive, force in jaws.PHASES
        ]
        return specs, worlds, assets, jaws.H, phases
    if case == "independent":
        spec, template, assets, _ = _wide(8)
        worlds = []
        for i in range(4):
            q, v = template.initial_state.q.copy(), template.initial_state.v.copy()
            # Translate both bodies together, then give each a different velocity.
            # Each pair remains in its own contact footprint. Neither side shares
            # the same stored coordinates across worlds even on the first step.
            q[0] += i * 0.001
            q[7] += i * 0.001
            v[0] += i * 0.002
            v[6] -= i * 0.001
            worlds.append(
                replace(template, world=7 + 1000 * i, initial_state=MuJoCoInitialState(q, v))
            )
        return (spec,), tuple(worlds), assets, H, [("independent", 3, None)]
    specs, template, assets = _cases()
    spec = specs[0]
    worlds = tuple(replace(template[0], world=7 + 1000 * i) for i in range(case))
    return (spec,), worlds, {spec.key: assets[spec.key]}, H, [("contact", 3, None)]


@pytest.mark.cuda
@pytest.mark.parametrize("case", [1, 4, 32, "jaws", "independent"])
def test_profile_complete_scene_steps(case, tmp_path, monkeypatch):
    import mujoco

    def reject_cpu_step(*args, **kwargs):
        pytest.fail("only the CUDA Scene may step")

    monkeypatch.setattr(mujoco, "mj_step", reject_cpu_step)
    specs, worlds, assets, h, phases = _workload(case)
    reference = []
    runs, uncached_runs = [], []
    output = tmp_path / "scene-step-profile.json"
    # The ordinary old/new comparison is counterbalanced around the two diagnostic runs.
    for run_index, variant in enumerate(
        ("uncached", "ordinary", "detailed", "detailed", "ordinary", "uncached")
    ):
        detailed = variant == "detailed"
        rows = []
        started = perf_counter_ns()
        with ExitStack() as stack:
            batch = stack.enter_context(
                MuJoCoTyxHeterogeneousBatch(
                    specs,
                    worlds,
                    mode="cuda",
                    resource_budget=MuJoCoTyxResourceBudget(DEVICE_PAYLOAD_LIMIT),
                )
            )
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
            ws = stack.enter_context(
                batch.prepare_device_workspace(include_bodies=True, include_area_pressure=True)
            )
            bucket_count = len({row.bucket for row in batch.layout.values()})
            nworld = len(worlds)
            assert bucket_count == (2 if case == "jaws" else 1)
            target = uncached_runs if variant == "uncached" else runs
            target.append(dict(detailed=detailed, setup_ns=perf_counter_ns() - started, rows=rows))
            if variant == "uncached":
                patch = stack.enter_context(monkeypatch.context())
                patch.setattr(scene_geometry, "_build_geometry_inputs", uncached_inputs)
            for phase, count, ctrl in phases:
                if ctrl is not None:
                    batch.set_controls(dict.fromkeys(batch.world_keys, ctrl))
                for _ in range(count):
                    tick = len(rows) + 1
                    owners = {k: batch.area_pressure_configuration_id(k) for k in batch.world_keys}
                    counter = _validation_calls(monkeypatch) if detailed else nullcontext()
                    with counter as built:
                        accepted, row = measure_scene_step(scene, h, detailed=detailed)
                    start = perf_counter_ns()
                    frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                    refreshed = perf_counter_ns()
                    arrays = ws.readback(frame)
                    copied = perf_counter_ns()
                    states = batch.integration_snapshot()
                    finished = perf_counter_ns()
                    row.update(
                        tick=tick,
                        phase=phase,
                        refresh_ns=refreshed - start,
                        readback_ns=copied - refreshed,
                        snapshot_ns=finished - copied,
                    )
                    assert accepted == dict.fromkeys(batch.world_keys, True)
                    assert frame.ticks == (tick,) * nworld
                    if detailed:
                        calls = row["calls"]
                        for key in (
                            "poses_and_requests",
                            "surface_packet",
                            "cuda_row_assembly",
                            "cuda_friction_solve",
                            "device_advance_other",
                            "device_entry",
                        ):
                            assert calls[key] == bucket_count
                        pairs = sum(len(assets[w.model_key]) for w in worlds)
                        assert calls["mesh_validation"] == calls["intersection"] == pairs
                        row["mesh_builds"] = dict(built)
                        if case == "independent":
                            assert built == {"pressure": 4, "rigid": 4}
                        elif case == "jaws":
                            assert 1 <= built["pressure"] <= 2 and 2 <= built["rigid"] <= 4
                        else:
                            assert built == {"pressure": 1, "rigid": 1}
                        assert calls["force_publication"] == nworld
                        assert calls["receipt_checks"] >= 2 * nworld
                        assert sum(row["exclusive_ns"].values()) == row["scene_ns"]
                    samples = 0
                    total_load = 0.0
                    for world in worlds:
                        k = world.world
                        fields = states[k].last_step.raw_contact_fields["area_friction"]
                        for result, contact, catalog in zip(
                            scene.last_geometry[k], fields, frame.area_catalogs[k], strict=True
                        ):
                            stamp = result.surface.stamp
                            assert stamp == contact["stamp"] == catalog["stamp"]
                            assert stamp.configuration_id == owners[k] and stamp.world_id == str(k)
                            assert stamp.step_index == tick - 1
                            samples += len(contact["sample_ids"])
                            total_load += float(np.sum(contact["normal_force"]))
                    if case == "independent":
                        assert len({states[w.world].v[0] for w in worlds}) == len(worlds)
                        assert len({states[w.world].q[7] for w in worlds}) == len(worlds)
                    row.update(samples=samples, total_normal_n=total_load)
                    if case != "jaws" or phase == "load":
                        assert samples and total_load > 0
                    capture = (
                        batch.last_step,
                        scene.last_geometry,
                        states,
                        frame.area_catalogs,
                        arrays,
                    )
                    if run_index == 0:
                        reference.append(capture)
                    else:
                        _same(capture, reference[tick - 1], fresh_stamp=True)
                    rows.append(row)
            assert len(rows) == sum(n for _, n, _ in phases)
        output.write_text(
            json.dumps(
                dict(
                    case=case,
                    worlds=nworld,
                    buckets=bucket_count,
                    timestep_s=h,
                    runs=runs,
                    uncached_runs=uncached_runs,
                ),
                allow_nan=False,
            ),
            encoding="utf-8",
        )
