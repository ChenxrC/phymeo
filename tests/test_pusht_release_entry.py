from types import SimpleNamespace

import numpy as np
import pytest

from experiments.area_pressure import pusht_release_entry as entry


def test_public_free_state_preserves_original_float32_cells():
    q = np.array([0.0, 0.1, -0.2, 0.0, 0.02, 0.6, 0.0, 0.8, 0.0], dtype=np.float32)
    model = SimpleNamespace(
        nq=9, nv=8, joint=lambda _: SimpleNamespace(id=2), jnt_qposadr=np.array([0, 1, 2])
    )
    state = dict(q=q.astype(float).tolist(), v=[0.25] * 8, time=1.7000000807456672)
    initial = entry.public_release_initial(model, state)
    assert np.asarray(initial.q, np.float32).tobytes() == q.tobytes()
    assert np.linalg.norm(initial.q[5:]) == pytest.approx(1.0, abs=2e-15)
    assert initial.time == state["time"]


def test_release_seed_requires_the_saved_velocity_and_clock():
    expected = dict(q=[1.0], v=[2.0], time=1.25)
    run = SimpleNamespace(state=expected, driver=SimpleNamespace(last_q=np.array([1.0])))

    def snapshot(v=2.0, t=1.25, tick=0):
        return SimpleNamespace(q=np.array([1.0]), v=np.array([v]), time=t, tick=tick)

    entry.require_release_seed(run, snapshot())
    zero = SimpleNamespace(state=dict(q=[1.0], v=[-0.0], time=1.25), driver=run.driver)
    with pytest.raises(ValueError):
        entry.require_release_seed(zero, snapshot(0.0))
    for bad in (snapshot(0.0), snapshot(t=1.24), snapshot(tick=1)):
        with pytest.raises(ValueError):
            entry.require_release_seed(run, bad)
