"""Finite frozen FANUC pose: collision must not trigger an initialization trap.

The pose comes from the failed B1d trajectory. It is a collision regression,
not a stability qualification for that trajectory or its controller settings.
"""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from test_native_exclusions import connection_plan, fanuc_template


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_fanuc_finite_pose_convex_collision(nworld):
    import physolve_tyx
    from physolve_tyx._src.collision_driver import collision
    from tyx.compat import warp as wp

    template = fanuc_template()
    template = template.with_collision_exclusions(connection_plan(template))
    model = template.compilation.fresh_native_model()
    expected = {(0, 5), (1, 3), (1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (4, 6)}
    reference = json.loads(
        (Path(__file__).parent / "fixtures/fanuc_frozen_contacts.json").read_text(encoding="utf-8")
    )
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=binary)
    assert hashlib.sha256(binary).hexdigest() == reference["model_sha256"]
    data = mujoco.MjData(model)
    data.qpos[:] = reference["qpos"]
    mujoco.mj_forward(model, data)
    expected_order = np.asarray(reference["fields"]["geom"])
    with wp.ScopedDevice("cuda:0"):
        native_model = physolve_tyx.put_model(model)
        native_data = physolve_tyx.put_data(model, data, nworld=nworld, nconmax=128, njmax=256)
        collision(native_model, native_data)
        wp.synchronize_stream()
        count = int(native_data.nacon.numpy()[0])
        assert count == nworld * len(expected)
        pairs = native_data.contact.geom.numpy()[:count]
        worlds = native_data.contact.worldid.numpy()[:count]
        for world in range(nworld):
            assert {tuple(pair) for pair in pairs[worlds == world]} == expected
            rows = np.flatnonzero(worlds == world)
            order = [rows[np.flatnonzero(np.all(pairs[rows] == pair, axis=1))[0]]
                     for pair in expected_order]
            for field in ("dist", "pos", "frame"):
                np.testing.assert_allclose(
                    getattr(native_data.contact, field).numpy()[order],
                    reference["fields"][field], atol=1e-6, rtol=0, err_msg=field,
                )
        for field in ("dist", "pos", "frame"):
            assert np.all(np.isfinite(getattr(native_data.contact, field).numpy()[:count]))
