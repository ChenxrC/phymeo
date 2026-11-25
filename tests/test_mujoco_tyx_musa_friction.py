"""S4000 acceptance for applied X loads on a supported X/Z sliding sphere."""

import json
import os
import sys
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import test_mujoco_tyx_constraints as t4
from test_mujoco_tyx_runtime import compile_source

from phymeo.backends.mujoco_resource_budget import MuJoCoTyxResourceBudget
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.training.resident_stepper import PhysicsStepFailure

pytestmark = [
    pytest.mark.musa,
    pytest.mark.skipif(os.environ.get("PHYMEO_TEST_MUSA") != "1", reason="requires a MUSA GPU"),
]
LOAD_RATIOS = {4: 0.25, 91: 1.2}  # T4's subcritical and supercritical load cases, in units of mg
SETTLE, PUSH = 120, 80  # 1 ms steps
MU = 0.6


@pytest.fixture(scope="module")
def kernel_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("musa-friction-kernels")


def _xml(dimension):
    root = ET.fromstring(t4.slider_xml(dim=dimension, mass=1.0))
    body = root.find("./worldbody/body")
    body.remove(body.find("joint[@name='j1']"))  # keep X and Z, remove Y
    return ET.tostring(root, encoding="unicode")


def _binaries(cache):
    return {
        str(p.relative_to(cache)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(cache.rglob("kernel.so"))
    }


def _native_trace(mj, model, loads):
    result = {}
    for key, force in loads.items():
        data = mj.MjData(model)
        rows = []
        for tick in range(1, SETTLE + PUSH + 1):
            data.qfrc_applied[:] = [force if tick > SETTLE else 0.0, 0.0]
            # mj_step leaves the constraint evaluation at input and integrates
            # qpos/qvel to output. Keep the independent trajectory continuous.
            mj.mj_step(model, data)
            wrench = np.zeros(6)
            on_ball = np.zeros(3)
            ids = []
            if data.ncon:
                assert data.ncon == 1
                contact = data.contact[0]
                ids = [int(x) for x in contact.geom]
                mj.mj_contactForce(model, data, 0, wrench)
                on_ball = contact.frame.reshape(3, 3).T @ wrench[:3]
            rows.append(
                dict(
                    q=data.qpos.tolist(),
                    v=data.qvel.tolist(),
                    time=float(data.time),
                    ids=ids,
                    normal_tangents=wrench[:3].tolist(),
                    ball=on_ball.tolist(),
                    plane=(-on_ball).tolist(),
                    generalized=data.qfrc_constraint.tolist(),
                    native_rows=int(data.nefc),
                )
            )
        result[key] = rows
    return result


def _record(state, dimension):
    receipt = state.last_step
    assert receipt.backend == "mujoco-tyx-musa" and receipt.tick == state.tick
    assert receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
    assert receipt.contact_time == receipt.input_time < receipt.output_time == state.time
    assert len(receipt.contacts) <= 1
    ids, wrench, ball, plane = [], np.zeros(3), np.zeros(3), np.zeros(3)
    if receipt.contacts:
        c = receipt.contacts[0]
        assert c.dimension == dimension
        ids = list(map(int, c.geometry_ids))
        wrench = c.wrench_contact_frame[:3]
        if dimension == 3:
            ball, plane = c.force_on_geometry2_world, c.force_on_geometry1_world
        else:
            ball, plane = c.force_on_sphere_world, c.static_reaction_world
    return dict(
        q=state.q.tolist(),
        v=state.v.tolist(),
        time=state.time,
        ids=ids,
        normal_tangents=wrench.tolist(),
        ball=ball.tolist(),
        plane=plane.tolist(),
        generalized=receipt.qfrc_constraint.tolist(),
        constraint_rows=len(receipt.constraints),
    )


def _read_entry(runtime, tick, current):
    from phymeo.backends.mujoco_entry_checkpoint import _accepted_entry

    bank = _accepted_entry(runtime._device)
    records = {}
    for i, key in enumerate(runtime.world_keys):
        record = current[key]
        meta = bank["entry_meta"][i].tolist()
        assert meta[:3] == [1, tick - 1, tick]
        assert meta[3] == bool(record["ids"]) and meta[5] == 0
        ids = bank["entry_contact_ids"][i, 0].tolist()
        wrench = bank["entry_contact_data"][i, 0, 13:16].tolist()
        assert wrench == record["normal_tangents"]
        if record["ids"]:
            assert ids[2:4] == record["ids"]
        records[key] = dict(meta=meta, ids=ids, normal_tangents=wrench)
    return records


def _run(mj, dimension, monkeypatch, cache):
    from phymeo.backends.mujoco_tyx import _kernels

    xml = _xml(dimension)
    compilation = compile_source(xml)
    model = compilation.mj_model
    assert model.nq == model.nv == model.njnt == 2
    mg = float(model.body_mass[1] * -model.opt.gravity[2])
    loads = {key: ratio * mg for key, ratio in LOAD_RATIOS.items()}
    native = _native_trace(mj, model, loads)
    actual = {k: [] for k in loads}
    readbacks = {}
    budget = MuJoCoTyxResourceBudget(64 * 1024 * 1024)
    before_open = _binaries(cache)

    def forbid(*args, **kwargs):
        pytest.fail("MUSA physics or contact extraction called a host solver")

    with MuJoCoRuntime(
        compilation,
        mode="musa",
        world_keys=tuple(loads),
        cache_root=cache,
        contact_capacity=1,
        constraint_capacity=4,
        resource_budget=budget,
    ) as runtime:
        assert runtime.capabilities["gpu"] and runtime.capabilities["implementation"] == "tyx"
        initial = runtime.musa_device_report()
        assert initial["device_name"] and initial["arch"] and initial["backend"] == "musa"
        with monkeypatch.context() as guard:
            guard.setattr(mj, "mj_step", forbid)
            guard.setattr(mj, "mj_forward", forbid)
            guard.setattr(mj, "mj_contactForce", forbid)
            guard.setattr(type(_kernels.advance), "__call__", forbid)
            for tick in range(1, SETTLE + PUSH + 1):
                if tick == SETTLE + 1:
                    runtime.set_controls(
                        {k: MuJoCoControl([], qfrc_applied=[f, 0.0]) for k, f in loads.items()}
                    )
                assert runtime.step(model.opt.timestep) == dict.fromkeys(loads, True)
                states = runtime.integration_snapshot()
                current = {k: _record(states[k], dimension) for k in loads}
                for k, value in current.items():
                    actual[k].append(value)
                if tick == 2:
                    warm_cache = _binaries(cache)  # both destination banks exercised
                    warm = runtime.musa_device_report()
                if tick in (SETTLE, SETTLE + 1, SETTLE + PUSH):
                    readbacks[tick] = _read_entry(runtime, tick, current)
        final = runtime.musa_device_report()
        counters = dict(runtime.counters)
        assert final["live_payload_bytes"] == budget.snapshot()["current_bytes"] > 0
    assert budget.snapshot()["current_bytes"] == budget.snapshot()["reserved_bytes"] == 0
    end_cache = _binaries(cache)
    return dict(
        source_xml=xml,
        dim=dimension,
        gravity_load_N=mg,
        lateral_N=loads,
        step_s=float(model.opt.timestep),
        settle_steps=SETTLE,
        push_steps=PUSH,
        native=native,
        musa=actual,
        device_start=initial,
        device_warm=warm,
        device_end=final,
        binary_counts=dict(
            before_open=len(before_open), after_two_steps=len(warm_cache), at_end=len(end_cache)
        ),
        cached_before=before_open,
        cached_after_two_steps=warm_cache,
        cached_at_end=end_cache,
        counters=counters,
        entry_readbacks=readbacks,
    )


def _column(rows, key):
    return np.asarray([r[key] for r in rows])


def _verify(case):
    dimension = case["dim"]
    start, warm, end = (case[k] for k in ("device_start", "device_warm", "device_end"))
    total = SETTLE + PUSH
    # T4 executes seven constraint/physics kernels and two accepted-entry kernels.
    # T0 executes the constant-mass step, input state and legacy normal record.
    per_step = 9 if dimension == 3 else 3
    for report, steps in ((warm, 2), (end, total)):
        assert report["launched_kernels"] - start["launched_kernels"] == per_step * steps
        assert report["completed_kernels"] - start["completed_kernels"] == per_step * steps
    assert end["failure"] is None and end["release_error"] is None
    assert end["downloaded_bytes"] > warm["downloaded_bytes"] > start["downloaded_bytes"]
    assert end["stream_synchronizations"] > warm["stream_synchronizations"]
    assert end["device_allocations"] == warm["device_allocations"]
    assert case["cached_at_end"] == case["cached_after_two_steps"] and case["cached_at_end"]
    for k in LOAD_RATIOS:
        a, b = case["musa"][k], case["native"][k]
        assert [r["ids"] for r in a] == [r["ids"] for r in b]
        assert all(not r["ids"] or r["ids"] == [0, 1] for r in b)
        np.testing.assert_allclose(_column(a, "time"), _column(b, "time"), rtol=0, atol=2e-6)
        # Original T4 short-path state and force tolerances, SI units.
        np.testing.assert_allclose(_column(a, "q"), _column(b, "q"), rtol=0, atol=2e-4)
        np.testing.assert_allclose(_column(a, "v"), _column(b, "v"), rtol=0, atol=3e-3)
        for field in ("normal_tangents", "generalized", "ball", "plane"):
            np.testing.assert_allclose(_column(a, field), _column(b, field), **t4.FORCE_TOL)
        force = _column(a, "ball")
        np.testing.assert_allclose(force + _column(a, "plane"), 0, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            force[:, [0, 2]], _column(a, "generalized"), rtol=1e-5, atol=1e-5
        )
        assert np.all(force[:, 2] >= -1e-6)
        if dimension == 3:
            np.testing.assert_array_equal(_column(a, "constraint_rows"), _column(b, "native_rows"))
            assert np.all(np.abs(force[:, 0]) <= MU * force[:, 2] + 3e-4)
        else:
            np.testing.assert_array_equal(force[:, :2], 0)
            load = case["lateral_N"][k]
            h = case["step_s"]
            t = np.arange(1, PUSH + 1)
            # The horizontal DOF has unit mass and no damping, armature or dry friction.
            np.testing.assert_allclose(_column(a, "v")[SETTLE:, 0], load * h * t, rtol=0, atol=1e-5)
            np.testing.assert_allclose(
                _column(a, "q")[SETTLE:, 0], load * h * h * t * (t + 1) / 2, rtol=0, atol=1e-5
            )


def test_musa_static_and_sliding_against_normal_only(monkeypatch, tmp_path, kernel_cache):
    import mujoco as mj

    monkeypatch.setitem(sys.modules, "warp", None)
    monkeypatch.setitem(sys.modules, "mujoco_warp", None)
    cases = {}
    for label, dimension in (("friction", 3), ("normal-only", 1)):
        case = _run(mj, dimension, monkeypatch, kernel_cache)
        cases[label] = case
        # Save native and GPU trajectories before applying the numerical gates.
        (tmp_path / f"{label}.json").write_text(json.dumps(case, allow_nan=False), encoding="utf8")
        _verify(case)
    summary = {}
    for key, ratio in LOAD_RATIOS.items():
        dry, free = (cases[label]["musa"][key] for label in ("friction", "normal-only"))
        force = _column(dry, "ball")[SETTLE:]
        speed = _column(dry, "v")[:, 0]
        # Input velocity is the previous output. Work uses SI force*time*velocity.
        work = float(np.sum(force[:, 0] * speed[SETTLE - 1 : -1]) * cases["friction"]["step_s"])
        delta = dry[-1]["q"][0] - dry[SETTLE - 1]["q"][0]
        free_delta = free[-1]["q"][0] - free[SETTLE - 1]["q"][0]
        mean_fx, _, mean_normal = force[-20:].mean(axis=0)
        applied = cases["friction"]["lateral_N"][key]
        assert mean_normal > 0 and mean_fx < 0 and work < 0
        assert 0 <= delta < free_delta
        if ratio < MU:
            # A softened static contact permits creep.
            assert applied < MU * mean_normal
            assert abs(speed[-1]) < 0.25 * free[-1]["v"][0] and delta < 0.25 * free_delta
        else:
            assert applied > MU * mean_normal
            assert speed[-1] > 0.01 and delta > 0.003
            assert -mean_fx >= 0.9 * MU * mean_normal
        summary[key] = dict(
            applied_N=applied,
            measured_friction_bound_N=float(MU * mean_normal),
            friction_work_J=work,
            dx_m=delta,
            normal_only_dx_m=free_delta,
            vx_m_s=float(speed[-1]),
            normal_only_vx_m_s=free[-1]["v"][0],
            mean_final_force_N=force[-20:].mean(axis=0).tolist(),
        )
    (tmp_path / "friction-comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf8"
    )


def test_two_moving_branches_receive_opposing_contact_forces(tmp_path):
    model, _, after, _ = t4.run_path(
        t4.branch_xml(dim=3, shape="sphere"),
        "musa",
        tmp_path / "two-sided.jsonl",
        v=np.array([0.1, 0.15, -0.08, -0.12]),
        steps=16,
        tiny=False,
    )
    assert model.nv == 4
    assert any(all(body > 0 for body in c.body_ids) for c in after.last_step.contacts)
    reaction = after.last_step.qfrc_constraint
    assert np.linalg.norm(reaction[:2]) > 1e-4 and np.linalg.norm(reaction[2:]) > 1e-4


def test_four_friction_rows_exceed_three_row_capacity(kernel_cache):
    starts = {
        4: MuJoCoInitialState([0.0, 0.0], [0.0, 0.0]),
        91: MuJoCoInitialState([0.0, 0.02], [0.0, 0.0]),
    }
    compilation = compile_source(_xml(3))
    with MuJoCoRuntime(
        compilation,
        mode="musa",
        world_keys=tuple(starts),
        initial_states=starts,
        cache_root=kernel_cache,
        contact_capacity=1,
        constraint_capacity=3,
    ) as runtime:
        before = runtime.snapshot()
        assert [len(before[k].contacts) for k in starts] == [1, 0]
        old = runtime.musa_device_report()
        with pytest.raises(PhysicsStepFailure) as result:
            runtime.step(runtime.timestep)
        assert result.value.acceptance_known and result.value.accepted_worlds == dict.fromkeys(
            starts, False
        )
        assert "capacity" in str(result.value.__cause__)
        new = runtime.musa_device_report()
        assert (new["active_bank"], new["revision"]) == (old["active_bank"], old["revision"])
        assert new["completed_kernels"] > old["completed_kernels"]
        for key, state in runtime.integration_snapshot().items():
            assert (
                state.last_step is None
                and state.tick == before[key].tick
                and state.time == before[key].time
            )
            np.testing.assert_array_equal(state.q, before[key].q)
            np.testing.assert_array_equal(state.v, before[key].v)
