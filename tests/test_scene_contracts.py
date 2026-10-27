"""Typed boundaries reject silent field loss, mutation and unregistered state families."""

import importlib
from dataclasses import dataclass

import numpy as np
import pytest

from phymeo import (
    ParticleModel,
    ParticleState,
    RigidBodyInputs,
    RigidBodyModel,
    RigidBodyState,
    StepInputs,
)


def api():
    return importlib.import_module("phymeo.core.scene")


def registry_api():
    return importlib.import_module("phymeo.runtime.registry")


def scene_data(dtype=np.float64):
    a = api()
    model = ParticleModel(np.array([1, 0], dtype), np.zeros(3, dtype))
    scene = a.SceneModel((a.DomainBinding("cloth", "particles", model),))
    state = a.SceneState({"cloth": ParticleState(np.zeros((2, 3), dtype), np.zeros((2, 3), dtype))})
    inputs = a.SceneInputs({"cloth": StepInputs.zeros(2, dtype=dtype)})
    return scene, state, inputs, a.BackendContext("numpy", dtype)


@pytest.mark.parametrize(
    "field,value",
    [
        ("substeps", True),
        ("substeps", 0),
        ("position_iterations", 1.5),
        ("velocity_iterations", -1),
        ("tolerance", -1),
        ("tolerance", np.nan),
        ("reference_length", 0),
        ("reference_speed", np.inf),
        ("record_substeps", 1),
    ],
)
def test_invalid_settings_cannot_enter_a_solver(field, value):
    with pytest.raises((TypeError, ValueError)):
        api().StepSettings(**{field: value})


def test_model_identity_detects_array_mutation_and_domain_mapping_is_fixed():
    scene, state, inputs, context = scene_data()
    scene.assert_unchanged()
    model = scene.domains[0].model
    model.inverse_mass.flags.writeable = True
    model.inverse_mass[0] = 2
    with pytest.raises(ValueError, match="changed"):
        scene.assert_unchanged()
    with pytest.raises(TypeError):
        state.domains["other"] = state.domains["cloth"]


def test_duplicate_and_unknown_domains_are_rejected_before_snapshot():
    a = api()
    scene, state, inputs, context = scene_data()
    with pytest.raises(ValueError):
        a.SceneModel((scene.domains[0], scene.domains[0]))
    with pytest.raises(ValueError):
        registry_api().StateRegistry().snapshot(
            a.SceneModel((a.DomainBinding("cloth", "unknown", scene.domains[0].model),)),
            state,
            inputs,
            context,
        )


@pytest.mark.parametrize(
    "target,keys", [("state", ()), ("inputs", ()), ("state", ("cloth", "extra"))]
)
def test_missing_or_extra_domains_are_never_silently_ignored(target, keys):
    a = api()
    scene, state, inputs, context = scene_data()
    if target == "state":
        state = a.SceneState({k: state.domains["cloth"] for k in keys})
    else:
        inputs = a.SceneInputs({k: inputs.domains["cloth"] for k in keys})
    with pytest.raises(ValueError):
        registry_api().StateRegistry().snapshot(scene, state, inputs, context)


def test_snapshot_owns_all_state_input_and_history_arrays_without_virtual_copy():
    a = api()
    scene, state, inputs, context = scene_data()

    class Borrowing(ParticleState):
        def copy(self):
            raise AssertionError("runtime must canonicalize registered fields")

    source = Borrowing(state.domains["cloth"].position, state.domains["cloth"].velocity)
    history = np.array([7.0])
    state = a.SceneState({"cloth": source}, {"warm": history})
    spec = (a.HistoryField("warm", (1,), np.float64),)
    owned, forces = registry_api().StateRegistry().snapshot(scene, state, inputs, context, spec)
    source.position[:] = 8
    inputs.domains["cloth"].force[:] = 9
    history[:] = 10
    np.testing.assert_array_equal(owned.domains["cloth"].position, np.zeros((2, 3)))
    np.testing.assert_array_equal(forces.domains["cloth"].force, np.zeros((2, 3)))
    np.testing.assert_array_equal(owned.history["warm"], [7])
    assert type(owned.domains["cloth"]) is ParticleState


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"warm": np.zeros(2)},
        {"warm": np.zeros(1, np.float32)},
        {"warm": np.array([np.nan])},
        {"warm": np.zeros(1), "extra": np.zeros(1)},
    ],
)
def test_bad_history_cannot_be_committed(values):
    a = api()
    scene, state, inputs, context = scene_data()
    spec = (a.HistoryField("warm", (1,), np.float64),)
    with pytest.raises((TypeError, ValueError)):
        registry_api().StateRegistry().snapshot(
            scene, a.SceneState(state.domains, values), inputs, context, spec
        )


@pytest.mark.parametrize(
    "change", ["dtype", "shape", "fixed_position", "fixed_velocity", "nonfinite"]
)
def test_candidate_preserves_every_registered_shape_precision_and_fixed_invariant(change):
    a = api()
    scene, state, inputs, context = scene_data()
    reg = registry_api().StateRegistry()
    before, _ = reg.snapshot(scene, state, inputs, context)
    candidate = ParticleState(np.zeros((2, 3)), np.zeros((2, 3)))
    if change == "dtype":
        candidate = ParticleState(np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32))
    elif change == "shape":
        candidate = ParticleState(np.zeros((3, 3)), np.zeros((3, 3)))
    elif change == "fixed_position":
        candidate.position[1, 0] = 1
    elif change == "fixed_velocity":
        candidate.velocity[1, 0] = 1
    else:
        candidate.velocity[0, 0] = np.nan
    with pytest.raises((TypeError, ValueError)):
        reg.validate_candidate(scene, before, a.SceneState({"cloth": candidate}), context)


def test_rigid_snapshot_and_validation_include_orientation_angular_velocity_and_torque():
    a = api()
    reg = registry_api().StateRegistry()
    model = RigidBodyModel(np.array([0.0]), np.zeros((1, 3)), np.zeros(3))
    body = RigidBodyState(
        np.zeros((1, 3)), np.array([[0.0, 0.0, 0.0, 1.0]]), np.zeros((1, 3)), np.zeros((1, 3))
    )
    forces = RigidBodyInputs(np.zeros((1, 3)), np.ones((1, 3)))
    scene = a.SceneModel((a.DomainBinding("body", "rigid_bodies", model),))
    before, owned_inputs = reg.snapshot(
        scene, a.SceneState({"body": body}), a.SceneInputs({"body": forces}), a.BackendContext()
    )
    forces.torque[:] = 20
    np.testing.assert_array_equal(owned_inputs.domains["body"].torque, np.ones((1, 3)))
    body.angular_velocity[0, 1] = 2
    with pytest.raises(ValueError):
        reg.validate_candidate(scene, before, a.SceneState({"body": body}), a.BackendContext())


def test_new_state_family_can_use_q_v_and_preserves_additional_physical_fields():
    a, r = api(), registry_api()

    @dataclass(frozen=True)
    class Model:
        mass: float = 2.0

    @dataclass
    class State:
        q: np.ndarray
        v: np.ndarray
        internal: np.ndarray

    @dataclass
    class Inputs:
        tau: np.ndarray

    def validate(model, state, inputs):
        if state.q.shape != (1,) or inputs.tau.shape != state.v.shape:
            raise ValueError("invalid scalar oscillator shape")

    family = r.StateFamily(
        "oscillator", Model, State, Inputs, ("q", "v", "internal"), ("tau",), "q", validate
    )
    registry = r.StateRegistry()
    registry.register(family)
    with pytest.raises(ValueError):
        registry.register(family)
    scene = a.SceneModel((a.DomainBinding("osc", "oscillator", Model()),))
    original = State(np.array([1.0]), np.array([2.0]), np.array([3.0]))
    owned, _ = registry.snapshot(
        scene,
        a.SceneState({"osc": original}),
        a.SceneInputs({"osc": Inputs(np.array([0.0]))}),
        a.BackendContext(),
    )
    original.internal[0] = 99
    np.testing.assert_array_equal(owned.domains["osc"].internal, [3])
    malformed = State(np.array([1.0]), np.array([2.0]), np.array([3.0, 4.0]))
    with pytest.raises(ValueError):
        registry.validate_candidate(
            scene, owned, a.SceneState({"osc": malformed}), a.BackendContext()
        )


def test_mixed_backend_precision_and_nonfinite_metrics_are_rejected():
    a = api()
    scene, state, inputs, context = scene_data()
    with pytest.raises(ValueError):
        registry_api().StateRegistry().snapshot(
            scene, state, inputs, a.BackendContext("numpy", np.float32)
        )
    with pytest.raises(ValueError):
        a.StageMetrics({"contact": float("nan")})
    with pytest.raises(ValueError):
        a.StageMetrics({"distance": -0.1})


def test_fingerprint_distinguishes_configuration_precision_and_order():
    a = api()
    x = np.array([1.0, 2.0])
    assert a.fingerprint({"x": x}) == a.fingerprint({"x": x.copy()})
    assert a.fingerprint(x) != a.fingerprint(x[::-1])
    assert a.fingerprint(x) != a.fingerprint(x.astype(np.float32))
    with pytest.raises(TypeError):
        a.fingerprint(object())
