"""Direct numerical entry points canonicalize state/input subclasses as well."""

import numpy as np
import pytest

from phymeo import (
    DistanceConstraints,
    ParticleModel,
    ParticleState,
    RigidBodyInputs,
    RigidBodyModel,
    RigidBodyState,
    RigidSemiImplicitSolver,
    RigidSpherePlaneSolver,
    SemiImplicitSolver,
    SphereShapes,
    StepInputs,
    TyxSemiImplicitSolver,
    TyxXPBDSolver,
    XPBDSolver,
)


def configuration(name):
    if name in ("particles", "xpbd"):
        model = ParticleModel(np.ones(1), np.zeros(3))
        state = ParticleState(np.array([[0.0, 1.0, 0.0]]), np.zeros((1, 3)))
        inputs = StepInputs(np.ones((1, 3)))
        solver = SemiImplicitSolver() if name == "particles" else XPBDSolver()
        fields, input_fields = ("position", "velocity"), ("force",)
    else:
        model = RigidBodyModel(np.ones(1), np.ones((1, 3)), np.zeros(3))
        state = RigidBodyState(
            np.array([[0.0, 1.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 1.0]]),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
        )
        inputs = RigidBodyInputs(np.ones((1, 3)), np.ones((1, 3)))
        solver = (
            RigidSemiImplicitSolver()
            if name == "rigid"
            else RigidSpherePlaneSolver(SphereShapes(np.array([0.2])))
        )
        fields = ("position", "orientation", "velocity", "angular_velocity")
        input_fields = ("force", "torque")

    class CachedState(type(state)):
        def copy(self):
            return self

    class CachedInputs(type(inputs)):
        def copy(self):
            return self

    state = CachedState(**{field: getattr(state, field) for field in fields})
    inputs = CachedInputs(**{field: getattr(inputs, field) for field in input_fields})
    return solver, model, state, inputs, fields, input_fields


@pytest.mark.parametrize("name", ["particles", "xpbd", "rigid", "sphere"])
def test_direct_solver_does_not_trust_virtual_state_copy(name):
    solver, model, state, inputs, fields, input_fields = configuration(name)
    before = {field: np.array(getattr(state, field), copy=True) for field in fields}
    before_inputs = {field: np.array(getattr(inputs, field), copy=True) for field in input_fields}
    result = solver.step(model, state, inputs, 0.1)
    assert result.report.accepted
    for field in fields:
        np.testing.assert_array_equal(getattr(state, field), before[field])
        assert not np.shares_memory(getattr(result.state, field), getattr(state, field))
    for field in input_fields:
        np.testing.assert_array_equal(getattr(inputs, field), before_inputs[field])


@pytest.mark.parametrize("name", ["particles", "xpbd", "rigid", "sphere"])
def test_direct_solver_revalidates_nonfinite_subclass_inputs(name):
    solver, model, state, inputs, _, _ = configuration(name)
    state.position[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solver.step(model, state, inputs, 0.1)


@pytest.mark.parametrize("name", ["particles", "xpbd", "rigid", "sphere"])
@pytest.mark.parametrize("part", ["state", "inputs"])
def test_direct_solver_rejects_cross_family_data_instead_of_dropping_fields(name, part):
    solver, model, state, inputs, _, _ = configuration(name)
    if name in ("particles", "xpbd"):
        if part == "state":
            state = RigidBodyState(
                state.position, np.array([[0.0, 0.0, 0.0, 1.0]]), state.velocity, np.zeros((1, 3))
            )
        else:
            inputs = RigidBodyInputs(inputs.force, np.ones((1, 3)))
    elif part == "state":
        state = ParticleState(state.position, state.velocity)
    else:
        inputs = StepInputs(inputs.force)
    with pytest.raises(TypeError, match="type"):
        solver.step(model, state, inputs, 0.1)


@pytest.mark.parametrize("name", ["particles", "xpbd", "rigid", "sphere"])
def test_numerical_failure_returns_an_owned_snapshot_even_for_subclasses(name):
    solver, model, state, inputs, fields, _ = configuration(name)
    state.velocity[:] = 1e308
    result = solver.step(model, state, inputs, 10.0)
    assert not result.report.accepted
    assert result.state is not state
    for field in fields:
        np.testing.assert_array_equal(getattr(result.state, field), getattr(state, field))
        assert not np.shares_memory(getattr(result.state, field), getattr(state, field))


@pytest.mark.tyx
@pytest.mark.parametrize("kind", ["free", "xpbd_prediction", "xpbd_alpha"])
def test_tyx_failure_snapshot_does_not_trust_subclass_copy(kind, tyx_mode, tmp_path):
    class CachedState(ParticleState):
        def copy(self):
            return self

    state = CachedState(np.ones((2, 3), np.float32), np.zeros((2, 3), np.float32))
    model, inputs = (
        ParticleModel(np.ones(2, np.float32), np.zeros(3)),
        StepInputs.zeros(2, dtype=np.float32),
    )
    options, dt = {}, 10.0
    if kind == "xpbd_alpha":
        options["constraints"] = DistanceConstraints(
            np.array([[0, 1]]), np.ones(1, np.float32), np.ones(1, np.float32)
        )
        dt = 1e-30
    else:
        state.velocity[:] = 2e38
    cls = TyxSemiImplicitSolver if kind == "free" else TyxXPBDSolver
    with cls(mode=tyx_mode, cache_root=tmp_path, **options) as solver:
        result = solver.step(model, state, inputs, dt)
    assert not result.report.accepted
    assert result.state is not state
    for field in ("position", "velocity"):
        np.testing.assert_array_equal(getattr(result.state, field), getattr(state, field))
        assert not np.shares_memory(getattr(result.state, field), getattr(state, field))
