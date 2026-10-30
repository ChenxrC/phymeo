"""Independent algebra controls; authored before any S2-L executable validation."""

from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

from phymeo.constraints.cloth_self_response import (SelfContactResponseLaw, SelfContactResponseLimits,
    SelfContactBudgets, PhaseFreeData, PhaseCapacities, propose_self_contact_phase,
    response_matrix, rank_witness, make_self_contact_load_plan)
from phymeo.core.capabilities import SolverFailure
from phymeo.core.model import ParticleModel
from phymeo.core.state import ParticleState
from phymeo.coupling.ports import ParticlePointPort
from phymeo.coupling.cloth_self_contact import SelfContactBinding

LAW = SelfContactResponseLaw(0.5, 0.5, normal_domain="unique_exposed_ray_v1")
LIMITS = SelfContactResponseLimits()
BUDGETS = SelfContactBudgets(0.0, 2.0**-40, 1e-8, 1e-8, 1e-8, 1e-8)


def explicit_binding(three=False, u=0.0):
    xy = np.array([[-1., -1.], [1., -1.], [0., 1.]])
    zs = (0., -.5, .5) if three else (.25, -.25)
    positions = np.vstack([np.column_stack((xy, np.full(3, z))) for z in zs])
    velocity = np.zeros_like(positions)
    if three:
        velocity[3:6, 2], velocity[6:9, 2] = 1., -1.
    else:
        velocity[:3] = (u/2, 0., -1.)
        velocity[3:] = (-u/2, 0., 1.)
    model = ParticleModel(np.ones(len(positions)), np.zeros(3))
    faces, weights, offsets, pairs, normals = [], [], [], [], []
    for fa, fb, sa, sb, normal in ((0, 1, -.25, .25, [0., 0., 1.]),) if not three else ((0, 1, -.25, .25, [0., 0., 1.]), (0, 2, .25, -.25, [0., 0., -1.])):
        for corner in range(3):
            row = len(faces)
            faces.extend((np.arange(3)+3*fa, np.arange(3)+3*fb))
            weights.extend((np.eye(3)[corner], np.eye(3)[corner]))
            offsets.extend((sa, sb))
            pairs.append((row, row+1))
            normals.append(normal)
    state = ParticleState(positions, velocity)
    port = ParticlePointPort(model, state, np.arange(len(positions)), domain="cloth")
    binding = SelfContactBinding(port, np.array(faces, np.int64), np.array(weights), np.array(offsets), np.array(pairs, np.int64))
    return model, state, binding, np.array(normals)


def solve(binding, normals, *, phase="terminal", closed=True, held=0.0):
    k = len(normals)
    caps = PhaseCapacities(np.full(k, closed, bool), np.ones(k, bool), np.full(k, held), tuple((i,) for i in range(k)))
    free = PhaseFreeData(binding.operator.relative_velocities(), np.zeros(k), np.zeros((k, 3)), 1/64, 1e-8)
    if phase == "held":
        free = replace(free, gaps=np.sum(normals*free.velocity, axis=1)/64)
    return propose_self_contact_phase(binding, normals, free, phase=phase, capacities=caps, law=LAW, limits=LIMITS)


@pytest.mark.parametrize("u,t,slip", [(35/64, -1/4, 0.), (35/16, -1/2, 35/32)])
def test_uniform_offset_tangent_coefficient_and_conditional_disk(u, t, slip):
    _, _, binding, normals = explicit_binding(u=u)
    w = response_matrix(binding)
    ex = np.tile([1., 0., 0.], 3)
    np.testing.assert_allclose((w @ ex).reshape(3, 3)[:, 0], 35/16, atol=1e-14, rtol=0)
    result = solve(binding, normals)
    np.testing.assert_allclose(result.normal_ns, 1., atol=1e-8, rtol=0)
    np.testing.assert_allclose(result.world_impulses[:, 0], t, atol=1e-8, rtol=0)
    final_u = binding.operator.relative_velocities() + (w @ result.world_impulses.ravel()).reshape(3, 3)
    np.testing.assert_allclose(final_u[:, 0], slip, atol=1e-8, rtol=0)


def test_shared_middle_face_has_negative_cross_response():
    _, _, binding, normals = explicit_binding(three=True)
    w = response_matrix(binding)
    nn = np.array([[normals[i] @ w[3*i:3*i+3, 3*j:3*j+3] @ normals[j] for j in range(6)] for i in range(6)])
    expected = np.block([[2*np.eye(3), -np.eye(3)], [-np.eye(3), 2*np.eye(3)]])
    np.testing.assert_array_equal(nn, expected)
    result = solve(binding, normals, phase="held")
    np.testing.assert_allclose(result.normal_ns, 1., atol=1e-8, rtol=0)
    plan = make_self_contact_load_plan(binding, result.world_impulses, phase="held")
    np.testing.assert_allclose(plan.node_impulses[:3], 0., atol=1e-8, rtol=0)
    bad = np.full(6, .5)
    np.testing.assert_array_equal(-np.ones(6)+nn @ bad, np.full(6, -.5))


def test_proposal_never_applies_base(monkeypatch):
    _, state, binding, normals = explicit_binding(u=35/64)
    original = state.velocity.copy()
    monkeypatch.setattr(binding.operator, "apply_impulses", lambda *_: pytest.fail("proposal applied state"))
    solve(binding, normals)
    np.testing.assert_array_equal(state.velocity, original)


def test_owned_capacity_is_not_duplicated_or_reused_between_phases():
    with pytest.raises(ValueError, match="duplicate"):
        PhaseCapacities(np.ones(2, bool), np.ones(2, bool), np.ones(2), ((0,), (0,)))
    _, _, binding, normals = explicit_binding(u=35/16)
    open_held = solve(binding, normals, phase="held", closed=False)
    np.testing.assert_array_equal(open_held.tangent_ns, np.zeros((3, 3)))
    assert np.all(open_held.capacity_ns == 0)


@pytest.mark.parametrize("matrix,accepted", [
    (np.array([[2., -1.], [-1., 2.]]), True),
    (np.array([[1., 1.], [1., 1.]]), False),
    (np.diag([1., 10.**12]), False),
    (np.diag([1., 2.**40]), False),
    (np.diag([1., 2.**39]), True),
])
def test_exact_full_rank_and_strict_condition_boundary(matrix, accepted):
    if accepted:
        receipt = rank_witness(matrix)
        assert F(*receipt[1]) < 10**12
    else:
        with pytest.raises(SolverFailure):
            rank_witness(matrix)


def test_asymmetry_is_retained_and_rejected():
    with pytest.raises(SolverFailure, match="asymmetry"):
        rank_witness(np.array([[2., .01], [0., 2.]]))


@pytest.mark.parametrize("change", [{"mu_dynamic": .25}, {"normal_domain": "first_axis"}, {"restitution": .1}])
def test_unsupported_laws_reject(change):
    with pytest.raises(ValueError):
        replace(LAW, **change)


def test_owned_plan_cannot_be_unfrozen_or_alias_caller():
    _, _, binding, _ = explicit_binding()
    world = np.ones((3, 3))
    plan = make_self_contact_load_plan(binding, world, phase="held")
    world[:] = 0
    assert np.all(plan.world_impulses == 1)
    with pytest.raises(ValueError):
        plan.world_impulses.flags.writeable = True


@pytest.mark.parametrize("bad", [np.zeros((2, 3), np.float32), np.zeros((2, 2)), np.full((2, 3), np.nan)])
def test_phase_data_rejects_shape_precision_and_nonfinite(bad):
    with pytest.raises((TypeError, ValueError)):
        PhaseFreeData(bad, np.zeros(2), np.zeros((2, 3)), 1/64, 1e-8)
