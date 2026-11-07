"""UNEXECUTED tests-first: real multi-owner tangent forces share one publication."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest


def _example():
    name = "phymeo_general_scene_friction_example"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "examples" / "general_scene_friction.py"
        spec = spec_from_file_location(name, path)
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_a01_public_multi_robot_tangent_accepts_real_bilateral_normals(tmp_path):
    # The old scene must first prove the physical 2 N / 2 N precondition.
    # RED is the existing public enable method's unsupported behavior, never a
    # missing new module or a test-generated substitute for normal contact.
    case_module = _example()
    with case_module.make_two_paddles(tmp_path) as case:
        scene = case.scene
        initial = scene.observe()
        events = scene.contact_events
        loads = scene.sample()
        assert len(loads.contacts) == 2
        (box_key,) = scene.body_keys[case.world, case_module.BOX]
        expected_pairs = {
            tuple(sorted((scene.shape_key(body, "tool", "ball"), box_key)))
            for body in (case_module.LOWER_TOOL, case_module.UPPER_TOOL)
        }
        assert {(load.first, load.second) for load in loads.contacts} == expected_pairs
        for load in loads.contacts:
            assert load.gap_m == pytest.approx(-1 / 1024, abs=2e-13)
            assert load.force_n == pytest.approx(2.0, abs=1e-10)
            assert len(load.sites) == 1
            site = load.sites[0]
            assert site.force_n == pytest.approx(2.0, abs=1e-10)
            assert site.fraction == pytest.approx(1.0, abs=1e-14)
            np.testing.assert_allclose(
                site.application_point_first_m,
                site.application_point_second_m,
                atol=1e-14,
                rtol=0,
            )
        box_row = scene.rigid_rows[case.world, case_module.BOX]
        np.testing.assert_allclose(loads.inputs.rigid.force[box_row], 0, atol=1e-12, rtol=0)
        np.testing.assert_allclose(loads.inputs.rigid.torque[box_row], 0, atol=1e-12, rtol=0)
        assert scene.contact_events is events
        assert scene.observe().step_number == initial.step_number == 0
        assert scene.tangent_history is None

        scene.enable_coupled_tangent(case.registry, stiffness_n_per_m=case_module.KT)

        assert scene.tangent_history.step_index == 0
        assert scene.tangent_history.active == ()
        assert scene.tangent_receipt is None
        assert scene.observe().time_s == 0.0
        assert scene.contact_events is events