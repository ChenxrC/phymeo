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
    result = dict(phase=plan.phase, plan_token=plan.binding_token, represented_nodes=plan.node_impulses.copy(),
                  final_velocity=state.velocity+model.inverse_mass[:, None]*plan.node_impulses,
                  backend="numpy", applications=1)
    if plan.phase == "held":
        result.update(original_force=np.zeros_like(plan.node_impulses), effective_force=plan.node_impulses.copy(),
                      predicted_velocity=state.velocity+model.inverse_mass[:, None]*plan.node_impulses,
                      h=1., gravity=np.zeros(3), material_scope="distance-material-equations-v1")
    return result


def retoken(plan, **changes):
    plan = replace(plan, **changes)
    names = ("phase", "domain", "triangles", "weights", "offsets", "pairs", "jacobians", "world_points", "world_impulses", "node_impulses", "raw_response")
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
    actual = receipt(plan, changed, model)
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


def test_r2_exact_on_range_delivery_and_off_range_residual():
    from fractions import Fraction as F
    from phymeo.validation.cloth_self_mapping import capacity_decomposition
    # One abstract 3-vector row over two unit masses: B=(I,-I), A=2I.
    b = [[F(int(i == j)) for j in range(3)]+[F(-int(i == j)) for j in range(3)] for i in range(3)]
    ell = [F(0), F(0), F(1), F(0), F(0), F(-1)]
    proof = capacity_decomposition(b, [F(1), F(1)], ell, [F(0)]*6)
    assert proof["effective"] == [0, 0, 1]
    assert proof["residual"] == [0]*6
    # Equal impulse on both nodes is orthogonal to B but is retained verbatim.
    epsilon = F(1, 2**32)
    perturbed = [value+epsilon*int(index in (0, 3)) for index, value in enumerate(ell)]
    changed = capacity_decomposition(b, [F(1), F(1)], perturbed, [F(0)]*6)
    assert changed["effective"] == [0, 0, 1]
    assert changed["residual"] == [epsilon, 0, 0, epsilon, 0, 0]


def test_r2_small_nodal_uncertainty_amplifies_beyond_phase_budget():
    from fractions import Fraction as F
    from phymeo.validation.cloth_self_mapping import capacity_decomposition
    from phymeo.validation.cloth_self_response import require_witness_phase_uncertainty
    # Original-form admission control, not a claimed triangle fixture:
    # B=diag(2^-16,1,1), A=diag(2^-32,1,1), kappa=2^32<10^12.
    # Native e=(2^-30,0,0) is below 1e-8 N*s; |L|e=(2^-14,0,0).
    b = [[F(1, 2**16), F(0), F(0)], [F(0), F(1), F(0)], [F(0), F(0), F(1)]]
    proof = capacity_decomposition(b, [F(1)], [F(0), F(0), F(1)], [F(1, 2**30), F(0), F(0)])
    assert F(*proof["rank"][1]) == 2**32
    assert proof["residual"] == [0, 0, 0]
    assert proof["residual_error"] == [0, 0, 0]
    assert proof["effective_error"] == [F(1, 2**14), 0, 0]
    witness = {"uncertainty_exact": tuple((v.numerator, v.denominator) for v in proof["effective_error"])}
    with pytest.raises(SolverFailure, match="amplification"):
        require_witness_phase_uncertainty(witness, 0, F(1), BUDGETS.velocity_mps)


def test_r2_changed_actual_delivery_cannot_reuse_unchanged_proposal_witness():
    from phymeo.validation.cloth_self_mapping import capacity_witness, verify_capacity_witness
    model, state, binding, _ = explicit_binding()
    plan = make_self_contact_load_plan(binding, np.tile([0., 0., 1.], (3, 1)), phase="terminal")
    actual = receipt(plan, state, model)
    witness = capacity_witness(plan, actual, model.inverse_mass, state.velocity, BUDGETS, original_positions=state.position)
    changed = dict(actual, final_velocity=actual["final_velocity"].copy())
    changed["final_velocity"][0, 2] += 2.**-32
    with pytest.raises(SolverFailure, match="actual delivery"):
        verify_capacity_witness(witness, plan, changed, model.inverse_mass, state.velocity, BUDGETS, original_positions=state.position)


def test_r2_exact_held_subtraction_and_native_prediction_uncertainty():
    from fractions import Fraction as F
    from phymeo.validation.cloth_self_mapping import actual_phase_impulse
    model, state, binding, _ = explicit_binding()
    plan = make_self_contact_load_plan(binding, np.zeros((3, 3)), phase="held")
    actual = receipt(plan, state, model)
    actual["predicted_velocity"] = actual["predicted_velocity"].copy()
    actual["predicted_velocity"][0, 0] += 2.**-32
    ell, uncertainty = actual_phase_impulse(plan, actual, model.inverse_mass, state.velocity)
    assert ell == [F(0)]*18
    assert uncertainty[0] == F(1, 2**32)
    assert all(value == 0 for value in uncertainty[1:])


def test_r2_open_birth_changed_held_phase_changes_only_its_owned_interval():
    from fractions import Fraction as F
    from phymeo.validation.cloth_self_mapping import capacity_decomposition
    b = [[F(int(i == j)) for j in range(3)]+[F(-int(i == j)) for j in range(3)] for i in range(3)]
    terminal = capacity_decomposition(b, [1, 1], [0, 0, 1, 0, 0, -1], [0]*6)
    held = capacity_decomposition(b, [1, 1], [0]*6, [0]*6)
    changed = capacity_decomposition(b, [1, 1], [0, 0, F(1, 4), 0, 0, F(-1, 4)], [0]*6)
    assert terminal["effective"][2] == 1 and held["effective"][2] == 0
    assert F(1, 2)*(terminal["effective"][2]+changed["effective"][2]) == F(5, 8)
    # Reusing the old combined capacity 1/2 would discard a changed phase receipt.
    assert terminal != changed and changed != held


def test_r2_open_birth_cannot_reuse_witness_after_one_held_receipt_changes():
    from phymeo.validation.cloth_self_mapping import capacity_witness, verify_capacity_witness
    model, state, binding, _ = explicit_binding()
    held = make_self_contact_load_plan(binding, np.zeros((3, 3)), phase="held")
    actual = receipt(held, state, model)
    witness = capacity_witness(held, actual, model.inverse_mass, state.velocity, BUDGETS, original_positions=state.position)
    changed = dict(actual)
    changed["effective_force"] = np.zeros((6, 3))
    changed["effective_force"][:3, 2] = 2.**-32
    changed["effective_force"][3:, 2] = -2.**-32
    changed["predicted_velocity"] = state.velocity+changed["effective_force"]
    changed["represented_nodes"] = changed["effective_force"].copy()
    with pytest.raises(SolverFailure, match="actual delivery"):
        verify_capacity_witness(witness, held, changed, model.inverse_mass, state.velocity, BUDGETS, original_positions=state.position)
