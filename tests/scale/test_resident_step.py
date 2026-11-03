"""Resident world dynamics, readback boundaries and keyed transactional state."""

from importlib import util

import numpy as np
import pytest


def resident_api():
    name = "phymeo.backends.tyx_world_batch"
    assert util.find_spec(name) is not None, "missing resident TYX world-step entry point"
    from phymeo.backends import tyx_world_batch

    return tyx_world_batch


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_free_ball_semi_implicit_trajectory(mode, tmp_path):
    api = resident_api()
    ball = api.FreeSphere(
        mass=2.0, radius=0.125, position=(0.0, 0.0, 2.0), gravity=(0.0, 0.0, -8.0)
    )
    with api.TyxWorldBatch([(31, ball), (4, ball)], mode, cache_root=tmp_path) as batch:
        for i in range(1, 33):
            assert batch.step(1 / 128) == {31: True, 4: True}
            states = batch.snapshot()
            for state in states.values():
                assert state.tick == i
                assert state.position == (0.0, 0.0, 2 - i * (i + 1) / 4096)
                assert state.velocity == (0.0, 0.0, -i / 16)
        np.testing.assert_equal(batch.summary()[4]["kinetic_energy_j"], 4.0)



def slider(*, delay=0, mass=2.0, gravity=(0.0, 0.0, 0.0), kp=16.0, kd=1.0, cap=4.0):
    import phymeo as ph
    from phymeo.solvers.driven_robot import DrivenRobotProgram

    model = ph.RobotModel(
        np.array([-1]), ("prismatic",), np.zeros((1, 3)), np.array([[0.0, 0, 0, 1]]),
        np.array([[1.0, 0, 0]]), np.array([mass]), np.zeros((1, 3)),
        np.ones((1, 3)), np.array(gravity),
    )
    program = DrivenRobotProgram(model, kp=[kp], kd=[kd], effort_limit=[cap], delay_steps=delay)
    initial = program.initial_state(ph.RobotState(np.zeros(1), np.zeros(1)))
    return program, initial


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_slider_commands_reversed_with_true_host_reference(mode, tmp_path):
    from contextlib import ExitStack

    api = resident_api()
    program, initial = slider()
    worlds = [(93, api.PrismaticSlider(program, initial)), (7, api.PrismaticSlider(program, initial))]
    with api.TyxWorldBatch(worlds, mode, cache_root=tmp_path) as batch, ExitStack() as stack:
        refs = {world: stack.enter_context(program.stepper()) for world, _ in worlds}
        states = {world: initial for world, _ in worlds}
        for i, target in enumerate((1, 1, -1, -1)):
            inputs = {w: program.inputs(np.array([sign * target], dtype=float)) for w, sign in ((93, 1), (7, -1))}
            batch.set_controls(inputs)
            assert batch.step(1 / 16) == {93: True, 7: True}
            observed = batch.snapshot()
            for w, sign in ((93, 1), (7, -1)):
                out = refs[w].step(states[w], inputs[w], 1 / 16)
                assert out.report.accepted
                states[w] = out.state
                effort = out.report.substeps[0].diagnostics["actuator_effort"][0]
                assert observed[w].applied_effort == (effort,)
                assert observed[w].applied_effort == (sign * target * 4,)
                assert observed[w].position == (sign * (1, 3, 4, 4)[i] / 128,)
                assert observed[w].velocity == (sign * (1, 2, 1, 0)[i] / 8,)
                np.testing.assert_allclose(observed[w].position, states[w].domains["robot"].q)
                np.testing.assert_allclose(observed[w].velocity, states[w].domains["robot"].v)


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_delayed_history_selected_restore_and_nonzero_replay(mode, tmp_path):
    from contextlib import ExitStack

    api = resident_api()
    program, initial = slider(delay=1, kp=8.0, kd=2.0, cap=3.0)
    ids = (81, 5)
    runs = [(w, api.PrismaticSlider(program, initial)) for w in ids]
    h = 1 / 8
    with api.TyxWorldBatch(runs, mode, cache_root=tmp_path) as batch, ExitStack() as stack:
        refs = {w: stack.enter_context(program.stepper()) for w in ids}
        old = dict.fromkeys(ids, initial)
        def advance(commands):
            batch.set_controls(commands)
            assert all(batch.step(h).values())
            result = batch.snapshot()
            for w in ids:
                reference = refs[w].step(old[w], commands[w], h)
                assert reference.report.accepted
                old[w] = reference.state
                qv = old[w].domains["robot"]
                np.testing.assert_allclose(result[w].position, qv.q, atol=2e-6)
                np.testing.assert_allclose(result[w].velocity, qv.v, atol=2e-6)
                np.testing.assert_allclose(result[w].history, old[w].history["robot/drive_queue"].ravel(), atol=2e-6)
                np.testing.assert_allclose(result[w].applied_effort, reference.report.substeps[0].diagnostics["actuator_effort"], atol=2e-6)
            return result
        def packets(values):
            return {w: program.inputs(np.array([target]), np.array([speed]), feedforward=np.array([ff])) for w, (target,speed,ff) in zip(ids,values, strict=True)}
        advance(packets(((0.5,0.25,2.0),(-0.75,-0.5,-3.0))))
        advance(packets(((-0.5,-0.25,0.375),(0.25,-1.0,1.0))))
        middle = batch.snapshot([81])
        assert tuple(middle) == (81,) and middle[81].velocity != (0.0,)
        old_a = old[81]
        command = packets(((0.125,0.25,0.5),(-0.25,-0.125,-0.25)))
        original = advance(command)
        batch.restore(middle)
        old[81] = old_a
        now = batch.snapshot()
        assert now[81] == middle[81] and now[5] == original[5]
        again = advance(command)
        assert again[81] == original[81]
        assert again[5].tick == original[5].tick + 1
        batch.reset([81])
        reset = batch.snapshot()
        assert reset[81].tick == 0 and reset[81].position == (0.0,)
        assert reset[81].history == (0.0, 0.0, 0.0) and reset[5] == again[5]




@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_mixed_free_motion_matches_single_rigid_solver_and_moves_failed_world_independently(mode, tmp_path):
    import phymeo as ph

    api = resident_api()
    program, initial = slider()
    ball = api.FreeSphere(mass=3.0, radius=0.25, position=(1, -2, 4), velocity=(0.5, 0.25, -0.125), gravity=(1, -2, -4))
    bad = api.FreeSphere(mass=1, radius=0.5, position=(0, 0, 0), gravity=(0, 0, 0))
    inertia = 0.4 * ball.mass * ball.radius ** 2
    model = ph.RigidBodyModel(np.array([1 / ball.mass]), np.full((1, 3), inertia), np.array(ball.gravity))
    ref = ph.RigidBodyState(np.array([ball.position]), np.array([[0.,0.,0.,1.]]), np.array([ball.velocity]), np.zeros((1,3)))
    force = np.array([[0.75, -1.5, 3.0]])
    rinputs = ph.RigidBodyInputs(force, np.zeros((1,3)))
    solver = ph.RigidSemiImplicitSolver()
    with api.TyxWorldBatch([(17, bad), (1, api.PrismaticSlider(program, initial)), (44, ball)], mode, cache_root=tmp_path) as batch:
        batch.set_controls({44: force[0], 17: (3e38, 0, 0)})
        before = batch.snapshot([17])
        assert batch.step(2) == {17: False, 1: True, 44: True}
        after = batch.snapshot()
        assert after[17] == before[17]
        out = solver.step(model, ref, rinputs, 2)
        assert out.report.accepted
        np.testing.assert_allclose(after[44].position, out.state.position[0])
        np.testing.assert_allclose(after[44].velocity, out.state.velocity[0])
        batch.set_controls({17: (1, 0, 0)})
        assert all(batch.step(1/8).values())
        state = batch.snapshot([17])[17]
        assert state.tick == 1 and state.position == (1/64,0.,0.)
        assert state.velocity == (1/8,0.,0.)


def test_warm_residency_timer_and_three_output_boundaries(tmp_path, monkeypatch):
    import json
    import tyx
    from tyx.backends import cuda_runtime

    api = resident_api()
    ball = api.FreeSphere(mass=2., radius=0.125, position=(0,0,2), gravity=(0,0,-8))
    worlds = [(w, ball) for w in (13, 2, 31, 7)]
    with api.TyxWorldBatch(worlds, "cuda", cache_root=tmp_path / "cache") as batch:
        assert len({array.__cuda_array_interface__["data"][0] for array in batch._states}) == 2
        assert len({array.__cuda_array_interface__["data"][0] for array in batch._ticks}) == 2
        batch.set_controls({2: (0.25, 1., 0.)})
        assert all(batch.step(1/128).values())
        checkpoint = batch.snapshot()
        with monkeypatch.context() as m:
            def unexpected_alloc(*args, **kwargs):
                pytest.fail("CUDA driver allocated device memory during the warm step window")
            m.setattr(cuda_runtime, "mem_alloc", unexpected_alloc)
            baseline = batch.stats()
            native0 = tyx.runtime_stats()
            for _ in range(32):
                assert all(batch.step(1/128).values())
            native1 = tyx.runtime_stats()
            after = batch.stats()
        for field in ("device_allocations", "topology_uploaded_bytes", "control_uploaded_bytes", "snapshot_downloaded_bytes", "summary_downloaded_bytes"):
            assert after[field] == baseline[field]
        assert after["completed_steps"] - baseline["completed_steps"] == 32
        assert after["status_downloaded_bytes"] - baseline["status_downloaded_bytes"] == 32 * 4 * len(worlds)
        assert native1.cuda_kernel_launches - native0.cuda_kernel_launches == 32
        assert native1.cuda_h2d_bytes == native0.cuda_h2d_bytes
        assert native1.cuda_d2h_bytes - native0.cuda_d2h_bytes == 32 * 4 * len(worlds)
        info = batch.summary()
        assert info[13]["kinetic_energy_j"] == pytest.approx((33/16)**2)
        summary_stats = batch.stats()
        assert 0 < summary_stats["summary_downloaded_bytes"] - after["summary_downloaded_bytes"]
        assert summary_stats["snapshot_downloaded_bytes"] == after["snapshot_downloaded_bytes"]
        group = batch.snapshot()
        snap_stats = batch.stats()
        assert snap_stats["snapshot_downloaded_bytes"] > after["snapshot_downloaded_bytes"]
        assert group[7].position[2] == 2 - 33*34/4096
        batch.restore({13: checkpoint[13]})
        assert batch.snapshot([13])[13] == checkpoint[13]
        assert batch.snapshot([2])[2] == group[2]
        evidence = dict(worlds=len(worlds), warm_steps=32, before=baseline, after=after, summary=summary_stats, snapshot=snap_stats, native_launches=native1.cuda_kernel_launches-native0.cuda_kernel_launches, native_h2d=native1.cuda_h2d_bytes-native0.cuda_h2d_bytes, native_d2h=native1.cuda_d2h_bytes-native0.cuda_d2h_bytes)
        (tmp_path / "resident-warm.json").write_text(json.dumps(evidence, indent=2), encoding="utf8")


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_invalid_commands_restore_and_interrupted_publish_preserve_previous_state(mode, tmp_path, monkeypatch):
    from dataclasses import replace

    api = resident_api()
    sphere = api.FreeSphere(mass=1., radius=0.25, position=(0,0,2))
    with pytest.raises(ValueError, match="duplicate"):
        api.TyxWorldBatch([(4,sphere), (4,sphere)], mode, cache_root=tmp_path)
    with api.TyxWorldBatch([(4,sphere), (19,sphere)], mode, cache_root=tmp_path) as batch:
        batch.set_controls({19: (1,0,0)})
        assert all(batch.step(1/8).values())
        before = batch.snapshot()
        with pytest.raises((ValueError, KeyError)):
            batch.set_controls({4:(2,0,0),47:(0,0,0)})
        with pytest.raises(ValueError):
            batch.restore({4:before[4],19:replace(before[19], position=(float("nan"),0.,0.))})
        assert batch.snapshot() == before
        with monkeypatch.context() as m:
            def late_failure(*args):
                raise RuntimeError("injected before committing the candidate")
            m.setattr(batch, "_read_status", late_failure)
            with pytest.raises(RuntimeError, match="injected"):
                batch.step(1/8)
        assert batch.snapshot() == before
        assert all(batch.step(1/8).values())
        result = batch.snapshot()
        assert result[4].tick == result[19].tick == 2
        assert result[19].position[0] == 3/64
        assert result[4].position[0] == 0




@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_tilted_prismatic_axis_uses_current_external_load_and_unsaturated_pd(mode, tmp_path):
    from dataclasses import replace
    import phymeo as ph

    api = resident_api()
    prototype, _ = slider()
    half = np.sqrt(0.5)
    model = replace(prototype.model, joint_orientation=np.array([[0.0,0.,half,half]]), gravity=np.array([0.,-2.,0.]))
    program = ph.DrivenRobotProgram(model, kp=[8.], kd=[2.], effort_limit=[100.])
    state = program.initial_state(ph.RobotState(np.array([0.125]), np.array([-0.25])))
    loads = ph.RobotInputs(np.array([0.25]), np.array([[17.,3.,-1.]]), np.array([[0.,-2.,1.]]))
    action = program.inputs(np.array([0.25]), np.array([0.5]), feedforward=np.array([0.5]), loads=loads)
    with api.TyxWorldBatch([(4,api.PrismaticSlider(program,state))],mode,cache_root=tmp_path) as batch, program.stepper() as runtime:
        batch.set_controls({4:action})
        assert batch.step(1/8) == {4:True}
        result = batch.snapshot()[4]
        ref = runtime.step(state, action, 1/8)
        assert ref.report.accepted
        info = ref.report.substeps[0].diagnostics
        assert 0 < abs(info["actuator_effort"][0]) < 100
        np.testing.assert_allclose(result.applied_effort, info["actuator_effort"], atol=2e-6)
        np.testing.assert_allclose(result.position, ref.state.domains["robot"].q, atol=2e-6)
        np.testing.assert_allclose(result.velocity, ref.state.domains["robot"].v, atol=2e-6)


def test_empty_and_unsupported_topology_before_device_allocation(tmp_path):
    api = resident_api()
    program, initial = slider(delay=2)
    with pytest.raises(ValueError, match="delay"):
        api.TyxWorldBatch([(7,api.PrismaticSlider(program,initial))], "interpret", cache_root=tmp_path)
    with api.TyxWorldBatch([], "interpret", cache_root=tmp_path) as batch:
        assert batch.step(1/128) == batch.snapshot() == batch.summary() == {}
        assert batch.stats()["device_allocations"] == 0
