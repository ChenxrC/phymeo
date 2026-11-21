from types import SimpleNamespace

import numpy as np
import pytest

from experiments.area_pressure.pusht_active_checkpoint import require_saved_state, selected_steps


def test_restore_preserves_the_archived_local_tick_clock_and_all_fp32_words():
    state = dict(local_tick=314, time=2.014000095659867, q=[-0.0, 0.5], v=[1.0])
    snapshot = SimpleNamespace(
        tick=314, time=state["time"], q=np.array(state["q"]), v=np.array([1.0])
    )
    require_saved_state(snapshot, state)
    snapshot.q[0] = 0.0
    with pytest.raises(ValueError):
        require_saved_state(snapshot, state)
    snapshot.q[0] = -0.0
    snapshot.tick = 0
    with pytest.raises(ValueError):
        require_saved_state(snapshot, state)


def test_replay_uses_only_published_commands_and_requires_consecutive_inputs():
    rows = [
        dict(
            selected=False,
            original_input_tick=9,
            restored=True,
            rollback_verification=dict(passed=True),
        ),
        dict(selected=True, original_input_tick=9, original_output_tick=10),
        dict(selected=True, original_input_tick=10, original_output_tick=11),
    ]
    assert list(selected_steps(rows, 9, 2)) == rows[1:]
    rows[0]["restored"] = False
    with pytest.raises(ValueError):
        list(selected_steps(rows, 9, 2))
    with pytest.raises(ValueError):
        list(selected_steps(rows[2:], 9, 1))
