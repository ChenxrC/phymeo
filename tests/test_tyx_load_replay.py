"""Replay changes input loads; actual tyx programs still compute all candidates."""

import numpy as np
import pytest
from test_load_replay import mixed_replay
from test_robot_dynamics import planar_mass

pytestmark = pytest.mark.tyx


def test_actual_robot_and_node_replay_impulse_once_and_reset(tyx_mode):
    request = {
        "robot": {"effort": np.array([0.02, -0.01])},
        "cloth": {"force": np.array([[0.0, 0, 0], [0.1, 0, 0]])},
    }
    expected_v = np.linalg.solve(planar_mass(np.array([0.2, -0.3])), [0.02, -0.01])
    with mixed_replay(tyx_mode) as replay:
        first = replay.evaluate(request)
        zero = replay.evaluate({})
        again = replay.evaluate(request)
        for trial in (first, zero, again):
            assert trial.result.report.accepted, trial.result.report.reason
            diag = trial.result.report.substeps[0].diagnostics
            assert diag["backend"] == "tyx:" + tyx_mode
            assert diag["execution"]["completed_launches"] > 0
            assert all(
                part["execution"]["completed_launches"] > 0
                for part in diag["participants"].values()
            )
        for trial in (first, again):
            np.testing.assert_allclose(
                trial.result.state.domains["robot"].v, expected_v, atol=2e-5, rtol=2e-5
            )
            np.testing.assert_allclose(
                trial.result.state.domains["robot"].q,
                np.array([0.2, -0.3]) + 0.05 * expected_v,
                atol=2e-5,
                rtol=2e-5,
            )
            np.testing.assert_allclose(
                trial.result.state.domains["cloth"].velocity, [[0, 0, 0], [0.1, 0, 0]], atol=2e-5
            )
            np.testing.assert_allclose(
                trial.result.state.domains["cloth"].position, [[0, 0, 0], [1.005, 0, 0]], atol=2e-5
            )
        np.testing.assert_allclose(zero.result.state.domains["robot"].q, [0.2, -0.3], atol=2e-5)
        np.testing.assert_array_equal(
            again.result.state.domains["robot"].q, first.result.state.domains["robot"].q
        )
        np.testing.assert_array_equal(replay.snapshot().domains["cloth"].velocity, np.zeros((2, 3)))


def test_actual_material_replay_discards_prior_workspace_multipliers(tyx_mode):
    request = {"cloth": {"force": np.array([[0.0, 0, 0], [0.1, 0, 0]])}}
    with mixed_replay(tyx_mode, material=True) as replay:
        first = replay.evaluate(request)
        free = replay.evaluate({})
        again = replay.evaluate(request)
        alpha = 0.001 / 0.05**2
        displacement = 0.005 * alpha / (1 + alpha)
        for trial in (first, free, again):
            assert trial.result.report.accepted, trial.result.report.reason
            assert (
                trial.result.report.substeps[0].diagnostics["participants"]["cloth"]["execution"][
                    "completed_launches"
                ]
                > 0
            )
        np.testing.assert_allclose(
            first.result.state.domains["cloth"].position[1], [1 + displacement, 0, 0], atol=2e-5
        )
        np.testing.assert_allclose(
            first.result.state.domains["cloth"].velocity[1], [displacement / 0.05, 0, 0], atol=2e-5
        )
        np.testing.assert_allclose(
            free.result.state.domains["cloth"].position, [[0, 0, 0], [1, 0, 0]], atol=2e-5
        )
        np.testing.assert_array_equal(
            again.result.state.domains["cloth"].position,
            first.result.state.domains["cloth"].position,
        )
