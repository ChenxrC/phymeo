"""Qualification of the explicitly declared Panda connection rule, not raw URDF."""

import json

import mujoco
import numpy as np
import pytest
from test_native_exclusions import eligibility
from test_panda_collision_model import panda_connection_plan, panda_template

pytest_plugins = ["test_physolve_runtime"]


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_panda_explicit_connection_native_step_forward_reset(payload, nworld):
    original = panda_template()
    plan = panda_connection_plan(original)
    template = original.with_collision_exclusions(plan)
    failures, completed = [], []

    def check(phase, actual, expected, *, exact=False, exact_fields=()):
        # Continue finite numerical comparisons to measure every requested phase;
        # the final assertion fails on ANY mismatch. Never broaden tolerances.
        for name, value in actual.items():
            other = expected[name]
            assert value.shape == other.shape and np.isfinite(value).all()
            assert np.isfinite(other).all()
            use_exact = exact or name in exact_fields
            try:
                if use_exact:
                    np.testing.assert_array_equal(value, other, err_msg=f"{phase}: {name}")
                else:
                    np.testing.assert_allclose(value, other, rtol=0, atol=1e-6,
                                               err_msg=f"{phase}: {name}")
            except AssertionError:
                failures.append(dict(phase=phase, field=name, exact=use_exact,
                                     max_abs=float(np.max(np.abs(value - other))) if value.size else 0))
        completed.append(phase)

    with template.bind(nworld).create_runtime(payload=payload, nconmax=128, njmax=256) as runtime:
        with runtime._scope():
            model, cpu = runtime._model, runtime._cpu_model
            pairs = model.nxn_geom_pair.numpy()
            include = model.nxn_pairid.numpy()[:, 0] >= -1
            assert {tuple(row) for row in pairs[include]} == eligibility(cpu)
            assert eligibility(cpu) == eligibility(original.compilation.mj_model) - set(
                plan.native_geom_pairs
            )
            np.testing.assert_array_equal(model.nxn_geom_pair_filtered.numpy(), pairs[include])
            np.testing.assert_array_equal(model.nxn_pairid_filtered.numpy(),
                                          model.nxn_pairid.numpy()[include])
            initial = runtime.snapshot()
            data = mujoco.MjData(cpu)
            template.initial_state.apply(template.compilation, data)
            mujoco.mj_forward(cpu, data)
            reference = runtime._frontend.put_data(cpu, data, nworld=nworld,
                                                   nconmax=128, njmax=256)
            runtime._frontend.prepare_workspace(model, reference)
            handle = runtime._native._create_runtime(model, reference, "tyx", "tyx")
            try:
                runtime._native._prepare_runtime(handle)
                runtime._native._forward_runtime(handle)
                runtime._complete()

                def reference_state():
                    return {name: getattr(reference, name).numpy() for name in initial}

                check("initial", initial, reference_state())
                forces = np.broadcast_to(np.linspace(-.05, .05, cpu.nv, dtype=np.float32),
                                         (nworld, cpu.nv)).copy()
                forces *= np.arange(1, nworld + 1, dtype=np.float32)[:, None] / nworld
                for step in range(6):
                    reference.qfrc_applied.assign(forces, stream=runtime._stream)
                    runtime._native._step_runtime(handle)
                    runtime.step(qfrc_applied=forces)
                    check(f"step{step + 1}", runtime.snapshot(), reference_state())
                before = runtime.snapshot()
                assert np.all(np.any(before["qvel"] != initial["qvel"], axis=1))
                np.testing.assert_allclose(before["time"], .002 * 6, atol=1e-6, rtol=0)
                runtime._native._forward_runtime(handle)
                runtime.forward()
                forward = runtime.snapshot()
                check("forward", forward, reference_state())
                check("forward_state", {k: forward[k] for k in ("qpos", "qvel", "time")},
                      before, exact=True)
                mask = np.arange(nworld) % 2 == 0
                runtime.reset(mask)
                reset = runtime.snapshot()
                check("masked_unselected", {k: v[~mask] for k, v in reset.items()},
                      {k: v[~mask] for k, v in forward.items()}, exact=True)
                check("masked_selected", {k: v[mask] for k, v in reset.items()},
                      {k: v[mask] for k, v in initial.items()},
                      exact_fields=("qpos", "qvel", "time", "qfrc_applied", "ctrl"))
                runtime.reset()
                check("full_reset", runtime.snapshot(), initial,
                      exact_fields=("qpos", "qvel", "time", "qfrc_applied", "ctrl"))
                for _ in range(6):
                    runtime.step(qfrc_applied=forces)
                check("replay", {k: runtime.snapshot()[k] for k in ("qpos", "qvel", "time")},
                      before, exact=True)
            finally:
                runtime._complete()
                runtime._native._close_runtime(handle, True)
    print("PANDA_RULE_RESULT " + json.dumps(dict(nworld=nworld, completed=completed,
                                                failures=failures)), flush=True)
    assert not failures, json.dumps(failures, indent=2)
