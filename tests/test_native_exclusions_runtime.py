"""FANUC with a declared connection exclusion, not a changed global default."""

import numpy as np
import pytest
from test_native_exclusions import connection_plan, eligibility, fanuc_template

pytest_plugins = ["test_physolve_runtime"]


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_explicit_fanuc_step_forward_reset(payload, nworld):
    original = fanuc_template()
    plan = connection_plan(original)
    template = original.with_collision_exclusions(plan)
    with template.bind(nworld).create_runtime(payload=payload, nconmax=128, njmax=256) as runtime:
        # The production frontend builds both tables from the same native input.
        model = runtime._model
        all_pairs = model.nxn_geom_pair.numpy()
        include = model.nxn_pairid.numpy()[:, 0] >= -1
        assert {tuple(row) for row in all_pairs[include]} == eligibility(
            template.compilation.mj_model
        )
        np.testing.assert_array_equal(model.nxn_geom_pair_filtered.numpy(), all_pairs[include])
        np.testing.assert_array_equal(
            model.nxn_pairid_filtered.numpy(), model.nxn_pairid.numpy()[include]
        )
        assert eligibility(template.compilation.mj_model) == eligibility(
            original.compilation.mj_model
        ) - set(plan.native_geom_pairs)
        initial = runtime.snapshot()
        force = np.full((nworld, template.compilation.mj_model.nv), 0.03, dtype=np.float32)
        for _ in range(4):
            runtime.step(qfrc_applied=force)
        after = runtime.snapshot()
        assert np.all(np.any(after["qvel"] != initial["qvel"], axis=1)), (
            "robot failed the existing motion check"
        )
        for name in ("qpos", "qvel"):
            np.testing.assert_allclose(
                after[name], np.broadcast_to(after[name][0], after[name].shape), atol=1e-6, rtol=0
            )
        mask = np.arange(nworld) % 2 == 0
        runtime.reset(mask)
        reset = runtime.snapshot()
        for field in after:
            np.testing.assert_array_equal(reset[field][~mask], after[field][~mask], err_msg=field)
            np.testing.assert_array_equal(reset[field][mask], initial[field][mask], err_msg=field)
        runtime.forward()
        forward = runtime.snapshot()
        for field in ("qpos", "qvel", "time"):
            np.testing.assert_array_equal(forward[field], reset[field])
        runtime.reset()
        for _ in range(4):
            runtime.step(qfrc_applied=force)
        for field in ("qpos", "qvel", "time"):
            np.testing.assert_allclose(runtime.snapshot()[field], after[field], atol=1e-6, rtol=0)
        runtime.step(qfrc_applied=force)
        uninterrupted = runtime.snapshot()
        runtime.reset()
        for _ in range(4):
            runtime.step(qfrc_applied=force)
        runtime.reset(mask)
        runtime.step(qfrc_applied=force)
        continued = runtime.snapshot()
        for field in uninterrupted:
            np.testing.assert_allclose(
                continued[field][~mask],
                uninterrupted[field][~mask],
                atol=1e-6,
                rtol=0,
                err_msg=field,
            )
        print("FANUC explicit-rule", nworld, "qvel[0]", after["qvel"][0].tolist())
