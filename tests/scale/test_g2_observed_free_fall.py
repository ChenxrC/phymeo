"""Engine-independent validation of the frame used for actual cross-solver timing."""

import numpy as np
import pytest
from g2_cross_engine_steps import ApplicationOutput, setup_rows


def test_registration_order_and_post_step_state_are_observable():
    keys, z, v = setup_rows(2, True)
    app = ApplicationOutput(tuple(reversed(keys)), keys, z, v)
    p = np.zeros((2, 3), np.float32)
    speed = p.copy()
    p[:, 2] = z[::-1]
    speed[:, 2] = v[::-1]
    observed = app.read(0, p, speed)
    assert app.keys == tuple(sorted(keys)) and observed.flags.c_contiguous
    assert observed[0, 2] != observed[1, 2] and observed[0, 5] != observed[1, 5]
    with pytest.raises(AssertionError, match="trajectory mismatch"):
        app.read(1, p, speed)
    with pytest.raises(ValueError, match="exactly one"):
        ApplicationOutput((keys[0], keys[0]), keys, z, v)
    with pytest.raises(ValueError, match="exactly one"):
        ApplicationOutput((*keys, keys[0]), keys, z, v)
