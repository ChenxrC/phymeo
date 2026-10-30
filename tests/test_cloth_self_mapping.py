"""Independent Jacobian, moment, work and actual-output forgery controls."""

from dataclasses import replace

import numpy as np
import pytest

from phymeo.constraints.cloth_self_response import make_self_contact_load_plan
from phymeo.core.capabilities import SolverFailure
from phymeo.core.scene import fingerprint
from phymeo.validation.cloth_self_mapping import SelfContactMappingLedger
from test_cloth_self_response import BUDGETS, explicit_binding


def receipt(plan, state, model):
    return dict(phase=plan.phase, plan_token=plan.binding_token, represented_nodes=plan.node_impulses.copy(),
                final_velocity=state.velocity+model.inverse_mass[:, None]*plan.node_impulses)


def retoken(plan, **changes):
    plan = replace(plan, **changes)
    names = ("phase", "domain", "triangles", "weights", "offsets", "pairs", "jacobians", "world_points", "world_impulses", "node_impulses")
    return replace(plan, binding_token=fingerprint({name: getattr(plan, name) for name in names}))


def test_offset_map_independent_b_oracle_and_aggregate_work():
    model, state, binding, _ = explicit_binding(u=35/64)
    world = np.tile([-.25, 0., 1.], (3, 1))
    plan = make_self_contact_load_plan(binding, world, phase="terminal")
    gradients = np.array([[-.5, -.25], [.5, -.25], [0., .5]])
    expected = np.zeros((6, 3))
    expected[:3, 0], expected[3:, 0] = -.25, .25
    expected[:3, 2] = 1 + 3*.25*(-.25)*gradients[:, 0]
    expected[3:, 2] = -1 + 3*.25*(-.25)*gradients[:, 0]
    np.testing.assert_allclose(plan.node_impulses, expected, atol=1e-14, rtol=0)
    report = SelfContactMappingLedger(model).require(plan, original_positions=state.position,
        original_velocity=state.velocity, actual_replay_or_terminal_receipt=receipt(plan, state, model), budgets=BUDGETS)
    assert report["actual_response_error_mps"] <= 1e-14
    assert report["virtual_work_error_j"] <= 1e-14


@pytest.mark.parametrize("omitted", ["both", "a", "b"])
def test_omitted_normal_derivative_is_caught_independently(omitted):
    model, state, binding, _ = explicit_binding()
    plan = make_self_contact_load_plan(binding, np.tile([-.25, 0., 1.], (3, 1)), phase="terminal")
    jac = plan.jacobians.copy()
    for attachment in range(len(jac)):
        if omitted == "both" or (attachment % 2 == 0) == (omitted == "a"):
            jac[attachment] = plan.weights[attachment, :, None, None]*np.eye(3)
    bad = retoken(plan, jacobians=jac)
    with pytest.raises(SolverFailure, match="map|impulse"):
        SelfContactMappingLedger(model).require(bad, original_positions=state.position,
            original_velocity=state.velocity, actual_replay_or_terminal_receipt=receipt(bad, state, model), budgets=BUDGETS)


def test_actual_terminal_double_application_rejected():
    model, state, binding, _ = explicit_binding()
    plan = make_self_contact_load_plan(binding, np.tile([0., 0., 1.], (3, 1)), phase="terminal")
    actual = receipt(plan, state, model)
    actual["final_velocity"] = state.velocity+2*plan.node_impulses
    with pytest.raises(SolverFailure, match="actual native"):
        SelfContactMappingLedger(model).require(plan, original_positions=state.position,
            original_velocity=state.velocity, actual_replay_or_terminal_receipt=actual, budgets=BUDGETS)


def test_separated_endpoint_couple_is_not_identically_zero():
    from phymeo.coupling.cloth_self_contact import SelfContactBinding
    from phymeo.coupling.ports import ParticlePointPort
    from phymeo.core.state import ParticleState
    model, state, old, _ = explicit_binding()
    positions = state.position.copy()
    positions[:3, 2] += .125
    changed = ParticleState(positions, state.velocity)
    base = ParticlePointPort(model, changed, np.arange(6), domain="cloth")
    binding = SelfContactBinding(base, old.triangles, old.weights, old.offsets, old.pairs)
    plan = make_self_contact_load_plan(binding, np.tile([.25, 0., 0.], (3, 1)), phase="held")
    actual = dict(phase="held", plan_token=plan.binding_token, represented_nodes=plan.node_impulses)
    report = SelfContactMappingLedger(model).require(plan, original_positions=positions,
        original_velocity=state.velocity, actual_replay_or_terminal_receipt=actual, budgets=BUDGETS)
    np.testing.assert_allclose(report["actual_couple_nms"], [0., 3/32, 0.], atol=1e-14, rtol=0)


@pytest.mark.parametrize("field", ["phase", "represented_nodes", "plan_token"])
def test_actual_mapping_receipt_forgery_rejects(field):
    model, state, binding, _ = explicit_binding()
    plan = make_self_contact_load_plan(binding, np.ones((3, 3)), phase="held")
    actual = receipt(plan, state, model)
    actual[field] = {"phase": "terminal", "represented_nodes": np.zeros((6, 3)), "plan_token": "stale"}[field]
    with pytest.raises(SolverFailure):
        SelfContactMappingLedger(model).require(plan, original_positions=state.position,
            original_velocity=state.velocity, actual_replay_or_terminal_receipt=actual, budgets=BUDGETS)
