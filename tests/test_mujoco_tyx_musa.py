"""Opt-in S4000 acceptance through the public MuJoCo solver entry point."""

import json
import os
import sys

import numpy as np
import pytest
from test_mujoco_tyx_runtime import SLIDER, compile_source, source

from phymeo.backends.mujoco_resource_budget import MuJoCoTyxResourceBudget
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.training.resident_stepper import PhysicsStepFailure

pytestmark = [
    pytest.mark.musa,
    pytest.mark.skipif(
        os.environ.get("PHYMEO_TEST_MUSA") != "1", reason="set PHYMEO_TEST_MUSA=1 on a MUSA GPU"
    ),
]


def test_musa_slider_against_native_at_every_step(monkeypatch, tmp_path):
    import mujoco as mj

    from phymeo.backends.mujoco_tyx import _kernels

    compilation = compile_source(SLIDER)
    model = compilation.mj_model
    controls = {4: 8.0, 91: -0.7}
    references = {key: mj.MjData(model) for key in controls}
    cpu_step = mj.mj_step
    history = []
    budget = MuJoCoTyxResourceBudget(8 * 1024 * 1024)

    def forbidden(*args, **kwargs):
        pytest.fail("MUSA physics called a CPU, Warp, or upstream interpreter execution path")

    monkeypatch.setitem(sys.modules, "mujoco_warp", None)
    monkeypatch.setitem(sys.modules, "warp", None)
    with MuJoCoRuntime(
        compilation,
        world_keys=tuple(controls),
        mode="musa",
        cache_root=tmp_path / "compile",
        resource_budget=budget,
    ) as runtime:
        assert runtime.capabilities["gpu"] and runtime.capabilities["implementation"] == "tyx"
        runtime.set_controls({key: MuJoCoControl([value]) for key, value in controls.items()})
        for tick in range(1, 25):
            for key, control in controls.items():
                references[key].ctrl[:] = control
                cpu_step(model, references[key])
            with monkeypatch.context() as guard:
                guard.setattr(mj, "mj_step", forbidden)
                guard.setattr(mj, "mj_forward", forbidden)
                guard.setattr(type(_kernels.advance), "__call__", forbidden)
                assert runtime.step(model.opt.timestep) == dict.fromkeys(controls, True)
            device = runtime.musa_device_report()
            assert device["active_bank"] == tick % 2
            assert device["completed_kernels"] == device["launched_kernels"] == 2 * tick
            for key, state in runtime.snapshot().items():
                ref = references[key]
                np.testing.assert_allclose(state.q, ref.qpos, rtol=0, atol=3e-5)
                np.testing.assert_allclose(state.v, ref.qvel, rtol=0, atol=3e-5)
                assert state.time == pytest.approx(ref.time, abs=2e-6)
                assert state.tick == tick and state.last_step.backend == "mujoco-tyx-musa"
                assert not state.last_step.contacts
                history.append(
                    dict(
                        tick=tick,
                        world=key,
                        q=state.q.item(),
                        v=state.v.item(),
                        q_ref=ref.qpos.item(),
                        v_ref=ref.qvel.item(),
                    )
                )

        assert references[4].qpos[0] > 0 > references[91].qpos[0]
        assert runtime.counters["completed_kernels"] == 48
        report = runtime.musa_device_report()
        assert report["backend"] == "musa" and report["device_name"] and report["arch"]
        assert report["live_payload_bytes"] == budget.snapshot()["current_bytes"] > 0
        result = dict(device=report, counts=dict(runtime.counters), history=history)
    assert budget.snapshot()["current_bytes"] == budget.snapshot()["reserved_bytes"] == 0
    result["bytes_after_close"] = budget.snapshot()["current_bytes"]
    (tmp_path / "slider.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf8"
    )


def test_failed_batch_keeps_last_published_state_and_recovers(monkeypatch, tmp_path):
    import mujoco as mj

    compilation = compile_source(SLIDER)
    model = compilation.mj_model
    controls = {4: 2.0, 91: -0.5}
    references = {key: mj.MjData(model) for key in controls}
    with MuJoCoRuntime(
        compilation, mode="musa", world_keys=tuple(controls), cache_root=tmp_path / "compile"
    ) as runtime:
        runtime.set_controls({key: MuJoCoControl([value]) for key, value in controls.items()})
        runtime.step(runtime.timestep)
        before = runtime.snapshot()
        report = runtime.musa_device_report()

        def failed_download():
            raise RuntimeError("injected failure reading the candidate bank")

        with monkeypatch.context() as fault:
            fault.setattr(runtime._device, "download", failed_download)
            with pytest.raises(PhysicsStepFailure) as error:
                runtime.step(runtime.timestep)
        assert error.value.accepted_worlds == dict.fromkeys(controls, False)
        after_failure = runtime.musa_device_report()
        assert after_failure["active_bank"] == report["active_bank"]
        assert after_failure["revision"] == report["revision"]
        assert after_failure["completed_kernels"] == report["completed_kernels"] + 2
        for key, state in runtime.snapshot().items():
            assert state.tick == before[key].tick and state.time == before[key].time
            np.testing.assert_array_equal(state.q, before[key].q)
            np.testing.assert_array_equal(state.v, before[key].v)
        assert runtime.step(runtime.timestep) == dict.fromkeys(controls, True)
        for key, state in runtime.snapshot().items():
            ref = references[key]
            ref.ctrl[:] = controls[key]
            mj.mj_step(model, ref, nstep=2)
            assert state.tick == 2
            np.testing.assert_allclose(state.q, ref.qpos, atol=3e-5, rtol=0)
            np.testing.assert_allclose(state.v, ref.qvel, atol=3e-5, rtol=0)


def _native_normal(mj, model, data):
    assert data.ncon in (0, 1)
    if not data.ncon:
        return 0.0
    wrench = np.zeros(6)
    mj.mj_contactForce(model, data, 0, wrench)
    return float(wrench[0])


def _drop_reference(mj, model, initial_q, steps):
    trajectory, endpoint = mj.MjData(model), mj.MjData(model)
    trajectory.qpos[:] = initial_q
    samples = []
    for _ in range(steps):
        # mj_step integrates the new q/v but leaves this step's input-stage
        # constraint forces in trajectory. Use separate data for the endpoint.
        mj.mj_step(model, trajectory)
        endpoint.qpos[:], endpoint.qvel[:], endpoint.time = (
            trajectory.qpos,
            trajectory.qvel,
            trajectory.time,
        )
        mj.mj_forward(model, endpoint)
        samples.append(
            dict(
                q=trajectory.qpos.item(),
                v=trajectory.qvel.item(),
                time=trajectory.time,
                entry_count=trajectory.ncon,
                entry_force=_native_normal(mj, model, trajectory),
                endpoint_count=endpoint.ncon,
                endpoint_force=_native_normal(mj, model, endpoint),
            )
        )
    return samples


def test_musa_drop_first_contact_and_sustained_support(monkeypatch, tmp_path):
    import mujoco as mj

    from phymeo.backends.mujoco_entry_checkpoint import _accepted_entry
    from phymeo.backends.mujoco_tyx import _kernels

    # Existing one-normal native-law fixture and its FP32 trajectory tolerance.
    # The last 40 ms begin over ten blended 13 ms solref time constants after
    # the later impact. Force uses the established 1% + 3 mN per-step bound.
    xml = source()
    compilation = compile_source(xml)
    model = compilation.mj_model
    h, steps, tail = model.opt.timestep, 240, 40
    q0 = {4: 0.007, 91: 0.021}  # 6 mm and 20 mm of clearance above the plane
    starts = {key: MuJoCoInitialState([q], [0.0]) for key, q in q0.items()}
    reference = {key: _drop_reference(mj, model, q, steps) for key, q in q0.items()}
    first_entry = {
        k: next(i for i, row in enumerate(rows, 1) if row["entry_count"])
        for k, rows in reference.items()
    }
    first_endpoint = {
        k: next(i for i, row in enumerate(rows, 1) if row["endpoint_count"])
        for k, rows in reference.items()
    }
    assert first_endpoint[4] < first_endpoint[91]
    assert all(first_entry[k] == first_endpoint[k] + 1 for k in q0)
    probe_ticks = {1, steps, *first_entry.values(), *first_endpoint.values()}
    mg = float(model.body_mass[1] * -model.opt.gravity[2])
    pair = tuple(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name) for name in ("floor", "ball"))
    budget = MuJoCoTyxResourceBudget(8 * 1024 * 1024)
    history, reads, plateau = [], [], {}

    def forbidden(*args, **kwargs):
        pytest.fail("MUSA integration or receipt used a host physics path")

    monkeypatch.setitem(sys.modules, "mujoco_warp", None)
    monkeypatch.setitem(sys.modules, "warp", None)
    with MuJoCoRuntime(
        compilation,
        mode="musa",
        world_keys=tuple(q0),
        initial_states=starts,
        contact_capacity=1,
        constraint_capacity=1,
        resource_budget=budget,
        cache_root=tmp_path / "compile",
    ) as runtime:
        assert runtime.capabilities["gpu"] and runtime.capabilities["possible_contact_pairs"] == 1
        assert runtime.capabilities["contact_dimensions"] == (1,)
        assert runtime.capabilities["domain"] == "single-slide-plane-sphere-normal-v1"
        initial_report = runtime.musa_device_report()
        assert (
            initial_report["backend"] == "musa"
            and initial_report["device_name"]
            and initial_report["arch"]
        )
        before = runtime.integration_snapshot()
        for tick in range(1, steps + 1):
            with monkeypatch.context() as guard:
                guard.setattr(mj, "mj_step", forbidden)
                guard.setattr(mj, "mj_forward", forbidden)
                guard.setattr(mj, "mj_contactForce", forbidden)
                guard.setattr(type(_kernels.advance), "__call__", forbidden)
                assert runtime.step(h) == dict.fromkeys(q0, True)
                after = runtime.integration_snapshot()
            if tick in probe_ticks:
                # snapshot() deliberately uses a detached CPU endpoint observer.
                # Inspect the separate accepted GPU contact bank as well.
                endpoints = runtime.snapshot()
                bank = _accepted_entry(runtime._device)
            for row_index, (key, state) in enumerate(after.items()):
                expected = reference[key][tick - 1]
                receipt = state.last_step
                assert state.tick == receipt.tick == tick
                assert receipt.backend == "mujoco-tyx-musa"
                assert receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
                assert receipt.contact_time == receipt.input_time == before[key].time
                assert (
                    receipt.output_time == state.time == pytest.approx(expected["time"], abs=2e-6)
                )
                np.testing.assert_array_equal(receipt.input_q, before[key].q)
                np.testing.assert_array_equal(receipt.input_v, before[key].v)
                np.testing.assert_allclose(
                    [state.q.item(), state.v.item()],
                    [expected["q"], expected["v"]],
                    rtol=0,
                    atol=3e-5,
                )
                contacts = receipt.contacts
                assert len(contacts) == expected["entry_count"]
                normal = 0.0
                if contacts:
                    assert contacts[0].active and contacts[0].dimension == 1
                    assert contacts[0].geometry_ids == pair
                    normal = float(contacts[0].wrench_contact_frame[0])
                    np.testing.assert_allclose(
                        contacts[0].force_on_sphere_world, [0.0, 0.0, normal], atol=1e-5
                    )
                if tick == first_entry[key]:
                    assert normal > 0
                np.testing.assert_allclose(
                    [normal, receipt.qfrc_constraint.item()],
                    expected["entry_force"],
                    rtol=0.01,
                    atol=0.003,
                )
                history.append(
                    dict(
                        world=key,
                        tick=tick,
                        q=state.q.item(),
                        v=state.v.item(),
                        input_time=receipt.input_time,
                        output_time=state.time,
                        entry_count=len(contacts),
                        entry_force=normal,
                        native=expected,
                    )
                )
                if tick in probe_ticks:
                    endpoint = endpoints[key]
                    assert endpoint.contact_phase == "endpoint-cpu-forward-recomputed"
                    assert endpoint.contact_time == state.time
                    assert len(endpoint.contacts) == expected["endpoint_count"]
                    end_force = sum(float(c.wrench_contact_frame[0]) for c in endpoint.contacts)
                    np.testing.assert_allclose(
                        end_force, expected["endpoint_force"], rtol=0.01, atol=0.003
                    )
                    meta = bank["entry_meta"][row_index]
                    np.testing.assert_array_equal(
                        meta, [1, tick - 1, tick, len(contacts), len(contacts), 0]
                    )
                    data = bank["entry_contact_data"][row_index, 0]
                    ids = bank["entry_contact_ids"][row_index, 0]
                    assert data[13] == normal
                    np.testing.assert_array_equal(bank["entry_q"][row_index, :1], receipt.input_q)
                    if contacts:
                        np.testing.assert_array_equal(ids[2:4], pair)
                        assert ids[0] == ids[1] == ids[7] == 1
                    else:
                        np.testing.assert_array_equal(ids, np.zeros_like(ids))
                    reads.append(
                        dict(
                            world=key,
                            tick=tick,
                            device_entry_meta=meta.tolist(),
                            device_contact_ids=ids.tolist(),
                            device_entry_force=float(data[13]),
                            endpoint_count=len(endpoint.contacts),
                            endpoint_force=end_force,
                        )
                    )
            before = after
        for key in q0:
            window = [row for row in history if row["world"] == key][-tail:]
            forces = [r["entry_force"] for r in window]
            assert all(r["entry_count"] == 1 and r["entry_force"] > 0.0 for r in window)
            assert max(abs(r["v"]) for r in window) <= 0.004
            plateau[key] = float(np.mean(forces))
            assert plateau[key] == pytest.approx(mg, rel=0.03)
        device = runtime.musa_device_report()
        launches = device["launched_kernels"] - initial_report["launched_kernels"]
        # Physics, input state, and plane/sphere contact record are three device
        # launches per batch, shared by both worlds even during free fall.
        assert (
            launches
            == device["completed_kernels"] - initial_report["completed_kernels"]
            == 3 * steps
        )
        assert device["downloaded_bytes"] > initial_report["downloaded_bytes"]
        assert device["stream_synchronizations"] > initial_report["stream_synchronizations"]
        assert device["live_payload_bytes"] == budget.snapshot()["current_bytes"] > 0
        assert device["failure"] is None and device["release_error"] is None
        result = dict(
            source_xml=xml,
            initial_q=q0,
            timestep=h,
            steps=steps,
            first_entry=first_entry,
            first_endpoint=first_endpoint,
            mg_N=mg,
            steady_mean_N=plateau,
            tolerances=dict(
                q_v_abs=3e-5,
                force_rel=0.01,
                force_abs_N=0.003,
                steady_mean_rel=0.03,
                steady_speed_m_s=0.004,
                steady_steps=tail,
            ),
            max_abs_error={
                field: max(abs(r[field] - r["native"][field]) for r in history)
                for field in ("q", "v", "entry_force")
            },
            device=device,
            device_start=initial_report,
            counters=dict(runtime.counters),
            entry_readbacks=reads,
            history=history,
        )
    result["bytes_after_close"] = budget.snapshot()["current_bytes"]
    assert result["bytes_after_close"] == budget.snapshot()["reserved_bytes"] == 0
    (tmp_path / "drop-contact.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf8"
    )


def test_contact_capacity_rejects_both_worlds_before_publication(tmp_path):
    starts = {4: MuJoCoInitialState([0.0], [0.0]), 91: MuJoCoInitialState([0.021], [0.0])}
    with MuJoCoRuntime(
        compile_source(source()),
        mode="musa",
        world_keys=tuple(starts),
        initial_states=starts,
        contact_capacity=0,
        constraint_capacity=1,
        cache_root=tmp_path / "compile",
    ) as runtime:
        before = runtime.snapshot()
        assert [len(before[k].contacts) for k in starts] == [1, 0]
        old = runtime.musa_device_report()
        with pytest.raises(PhysicsStepFailure) as result:
            runtime.step(runtime.timestep)
        assert result.value.acceptance_known and result.value.accepted_worlds == dict.fromkeys(
            starts, False
        )
        assert "contact_or_constraint_capacity" in str(result.value.__cause__)
        new = runtime.musa_device_report()
        assert (new["active_bank"], new["revision"]) == (old["active_bank"], old["revision"])
        assert new["completed_kernels"] > old["completed_kernels"]
        for key, state in runtime.integration_snapshot().items():
            assert (
                state.last_step is None
                and state.time == before[key].time
                and state.tick == before[key].tick
            )
            np.testing.assert_array_equal(state.q, before[key].q)
            np.testing.assert_array_equal(state.v, before[key].v)


@pytest.mark.parametrize(
    "xml",
    [
        pytest.param(source(condim=4), id="fixed-torsional"),
        pytest.param(source(free=True, condim=4), id="floating-torsional"),
        pytest.param(
            '<mujoco><default><geom contype="0" conaffinity="0"/></default><worldbody>'
            '<body><freejoint/><geom size=".1"/></body>'
            '<body><freejoint/><geom size=".1"/></body></worldbody></mujoco>',
            id="two-free-roots",
        ),
        pytest.param(
            '<mujoco><default><geom contype="0" conaffinity="0"/></default><worldbody>'
            '<body><freejoint/><geom size=".1"/></body>'
            '<body><joint type="hinge"/><geom size=".1"/></body></worldbody></mujoco>',
            id="free-and-independent-hinge",
        ),
    ],
)
def test_unported_mixed_models_are_rejected_before_device_allocation(monkeypatch, xml):
    from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime, MuJoCoTyxUnsupported

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported model reached device allocation")

    monkeypatch.setattr(MuJoCoTyxRuntime, "_build_device", forbidden)
    with pytest.raises(MuJoCoTyxUnsupported, match="musa-domain"):
        MuJoCoRuntime(compile_source(xml), mode="musa")


def test_joint_limit_overflow_preserves_last_accepted_batch(tmp_path):
    from test_mujoco_tyx_constraints import joint_xml

    # The upper-limit margin begins at 0.19 rad. Only world 4 enters it
    # during the first accepted step. Both worlds always have one dry row.
    starts = {4: MuJoCoInitialState([0.1895], [1.5]), 91: MuJoCoInitialState([0.0], [0.0])}
    compilation = compile_source(joint_xml("hinge", dry=0.2))
    for capacity in (1, 2):
        with MuJoCoRuntime(
            compilation,
            mode="musa",
            world_keys=tuple(starts),
            initial_states=starts,
            contact_capacity=0,
            constraint_capacity=capacity,
            cache_root=tmp_path / "compile",
        ) as runtime:
            assert all(runtime.step(runtime.timestep).values())
            before = runtime.integration_snapshot()
            assert before[4].q.item() > 0.19 and abs(before[91].q.item()) < 0.19
            for state in before.values():
                assert state.tick == 1 and [
                    r.constraint_type for r in state.last_step.constraints
                ] == [1]
                assert not state.last_step.contacts
            if capacity == 2:
                assert all(runtime.step(runtime.timestep).values())
                after = runtime.integration_snapshot()
                assert all(s.tick == 2 for s in after.values())
                assert [r.constraint_type for r in after[4].last_step.constraints] == [1, 3]
                assert [r.constraint_type for r in after[91].last_step.constraints] == [1]
                continue
            device = runtime.musa_device_report()
            with pytest.raises(PhysicsStepFailure) as result:
                runtime.step(runtime.timestep)
            assert result.value.acceptance_known
            assert result.value.accepted_worlds == dict.fromkeys(starts, False)
            assert "capacity" in str(result.value.__cause__)
            after = runtime.musa_device_report()
            assert (after["active_bank"], after["revision"]) == (
                device["active_bank"],
                device["revision"],
            )
            for key, state in runtime.integration_snapshot().items():
                assert (state.tick, state.time) == (before[key].tick, before[key].time)
                np.testing.assert_array_equal(state.q, before[key].q)
                np.testing.assert_array_equal(state.v, before[key].v)
                assert [r.constraint_type for r in state.last_step.constraints] == [1]
                np.testing.assert_array_equal(
                    state.last_step.qfrc_constraint, before[key].last_step.qfrc_constraint
                )
