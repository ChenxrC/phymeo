"""Actual tyx execution; no host substitute or fallback."""

import numpy as np
import pytest
from test_initial_robot_material import check_sliding, check_support, diag, material_case

pytestmark = pytest.mark.tyx


def test_actual_bounded_material_pair(tyx_mode):
    runtime, state, inputs = material_case(mode=tyx_mode)
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert result.report.accepted, result.report.reason
    d = diag(result)
    np.testing.assert_allclose(d["world_impulses"], [[0, 4 / 7, 0]], atol=5e-5, rtol=0)
    np.testing.assert_allclose(
        result.state.domains["nodes"].velocity[:, 1], [-2 / 7, 6 / 7], atol=5e-5, rtol=0
    )
    assert d["native_backend"] == "tyx:" + tyx_mode
    assert result.report.substeps[0].diagnostics["execution"]["completed_launches"] > 0


@pytest.mark.parametrize("h", [1 / 16, 1 / 32])
def test_actual_material_support_two_seconds(tyx_mode, h):
    check_support(tyx_mode, h)


@pytest.mark.parametrize("mu", [1.0, 0.1])
def test_actual_material_friction_step_refinement(tyx_mode, mu):
    coarse, _ = check_sliding(tyx_mode, 1 / 16, mu)
    fine, _ = check_sliding(tyx_mode, 1 / 32, mu)
    if mu == 0.1:
        assert fine <= 0.6 * coarse


def test_actual_late_material_evidence_rollback(tyx_mode):
    from test_initial_robot_material import check_late_trace_rollback

    check_late_trace_rollback(tyx_mode)


@pytest.mark.parametrize("coordinate", [1e-25, -1e-25])
def test_prismatic_translation_does_not_evaluate_unused_rotation(tyx_mode, coordinate):
    from phymeo.backends.tyx_robot_math import TyxRobotMath

    runtime, _, _ = material_case(mode=tyx_mode)
    model = runtime._program.model
    runtime.close()
    with TyxRobotMath(model, tyx_mode) as engine:
        with np.errstate(all="raise"):
            geometry = engine.geometry(np.array([coordinate]))
        np.testing.assert_array_equal(geometry.rotation, np.eye(3)[None])
        np.testing.assert_array_equal(geometry.position, [[0, float(np.float32(coordinate)), 0]])
