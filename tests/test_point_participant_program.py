"""Independent particle equations and public adapter ownership controls."""

from dataclasses import replace

import numpy as np
import pytest

from phymeo.constraints.distance import DistanceConstraints
from phymeo.core.capabilities import SolverFailure
from phymeo.core.model import ParticleModel
from phymeo.core.scene import (
    BackendContext, DomainBinding, HistoryField, SceneInputs, SceneModel,
    SceneState, StepSettings, fingerprint,
)
from phymeo.core.state import ParticleState, StepInputs
from phymeo.coupling.participant import ClothParticipant
from phymeo.runtime.load_replay import SceneLoadReplay
from phymeo.runtime.point_participant_program import PointParticipantProgram
from phymeo.runtime.scene_stepper import SceneStepper
from phymeo.runtime.terminal_impulse import PointVelocityImpulses, TerminalImpulseProgram


def particle_fixture(mode="numpy", *, material=False, domains=("cloth",), history=False):
    model = ParticleModel(np.array([0.5, 0.25]), np.zeros(3))
    constraint = DistanceConstraints(
        np.array([[0, 1]]), np.array([1.0]), np.array([1 / 16])
    ) if material else None
    scene = SceneModel(tuple(DomainBinding(name, "particles", model) for name in domains))
    schema = (HistoryField("counter", (1,), np.int64),) if history else ()
    state = SceneState(
        {name: ParticleState(np.array([[0., 0., 0.], [1., 0., 0.]]),
                             np.array([[.25, 0., 0.], [-.125, 0., 0.]])) for name in domains},
        {"counter": np.array([7], dtype=np.int64)} if history else {},
    )
    inputs = SceneInputs({name: StepInputs(np.array([[.5, 0., 0.], [0., 0., 0.]])) for name in domains})
    if mode == "numpy":
        factories = {name: ClothParticipant(model, constraint, record_material_path=True) for name in domains}
    else:
        from phymeo.backends.tyx_participants import TyxClothParticipant
        factories = {name: TyxClothParticipant(model, constraint, mode, record_material_path=True) for name in domains}
    program = PointParticipantProgram(scene, factories, history_fields=schema)
    settings = StepSettings(substeps=1, position_iterations=512, tolerance=1e-8 if mode == "numpy" else 2e-5)
    context = BackendContext(program.capabilities.backend)
    return scene, state, inputs, program, settings, context


def terminal_requests(domain="cloth"):
    return (
        PointVelocityImpulses(domain, np.array([0]), np.array([[1., 0., 0.]])),
        PointVelocityImpulses(domain, np.array([0, 1]), np.array([[-.25, 0., 0.], [.5, 0., 0.]])),
    )


def terminal_trial(mode="numpy", *, material=False, history=False):
    scene, state, inputs, base, settings, context = particle_fixture(mode, material=material, history=history)
    wrapper = TerminalImpulseProgram(base, terminal_requests(), registry=base.registry)
    with SceneStepper(scene, wrapper, settings, context, base.registry) as runtime:
        result = runtime.step(state, inputs, .25)
    return state, inputs, base, result


@pytest.mark.parametrize("material", [False, True])
def test_terminal_duplicate_rows_match_independent_particle_equations(material):
    before, _, base, result = terminal_trial(material=material, history=True)
    assert result.report.accepted
    after = result.state.domains["cloth"]
    expected_x = [3 / 64, 63 / 64] if material else [5 / 64, 31 / 32]
    expected_v = [9 / 16, 1 / 16] if material else [11 / 16, 0.]
    np.testing.assert_allclose(after.position[:, 0], expected_x, rtol=0, atol=1e-14)
    np.testing.assert_allclose(after.velocity[:, 0], expected_v, rtol=0, atol=1e-14)
    receipt = result.report.substeps[0].diagnostics["terminal_impulse"]
    assert receipt["applications"] == 1 and len(receipt["batches"]) == 1
    pre = receipt["pre_impulse_state"].domains["cloth"]
    np.testing.assert_array_equal(after.position, pre.position)
    ell = np.array([[.75, 0., 0.], [.5, 0., 0.]])
    inverse = base.participants["cloth"].model.inverse_mass
    expected_work = np.sum(ell * pre.velocity) + .5 * np.sum(inverse[:, None] * ell**2)
    actual_work = .5 * np.sum((after.velocity**2 - pre.velocity**2) / inverse[:, None])
    assert actual_work == pytest.approx(expected_work, abs=1e-14)
    assert before.history["counter"].tobytes() == result.state.history["counter"].tobytes()
    result.state.history["counter"][0] = 9
    assert before.history["counter"][0] == 7
    assert "robot" not in result.state.domains


def test_replay_p_zero_p_retains_original_history_and_borrowed_factory():
    scene, original, inputs, base, settings, context = particle_fixture(history=True)
    signature = fingerprint((original, inputs))
    with SceneLoadReplay(scene, base, original, inputs, .25, channels={"cloth": ("force",)},
                         settings=settings, context=context, registry=base.registry) as replay:
        pulse = {"cloth": {"force": np.array([[.5, 0., 0.], [0., 0., 0.]])}}
        first, zero, second = replay.evaluate(pulse), replay.evaluate({}), replay.evaluate(pulse)
    assert first.result.report.accepted and zero.result.report.accepted and second.result.report.accepted
    assert fingerprint(first.result.state) == fingerprint(second.result.state)
    assert fingerprint((original, inputs)) == signature
    np.testing.assert_array_equal(first.result.state.domains["cloth"].velocity[:, 0], [9 / 16, -1 / 8])
    np.testing.assert_array_equal(zero.result.state.domains["cloth"].velocity[:, 0], [5 / 16, -1 / 8])
    with SceneStepper(scene, base, settings, context, base.registry) as runtime:
        again = runtime.step(original, inputs, .25)
    assert again.report.accepted
    assert fingerprint(again.state) == fingerprint(zero.result.state)


class ObservedFactory:
    """Test-only public-operation wrapper, never a production replacement solver."""

    def __init__(self, factory, events, name, hook=None, fail_create=False):
        self.factory, self.events, self.name = factory, events, name
        self.hook, self.fail_create = hook, fail_create
        self.model, self.family = factory.model, factory.family
        self.capabilities, self.point_role = factory.capabilities, factory.point_role

    def problem_signature(self):
        return self.factory.problem_signature()

    def create_workspace(self, *args):
        if self.fail_create:
            raise SolverFailure("injected construction failure")
        child = self.factory.create_workspace(*args)
        self.events.append((self.name, "create"))
        owner = self
        class PublicWrapper:
            def __getattr__(self, name):
                operation = getattr(child, name)
                def call(*arguments):
                    owner.events.append((owner.name, name))
                    result = operation(*arguments)
                    if owner.hook is not None:
                        owner.hook(name)
                    return result
                return call
        return PublicWrapper()


def observed_program(base, events, hooks=None, failed_domain=None):
    return PointParticipantProgram(base.scene, {
        name: ObservedFactory(factory, events, name, (hooks or {}).get(name), name == failed_domain)
        for name, factory in base.participants.items()
    }, history_fields=base.history_fields)


def prepare(base, state, inputs, settings, context):
    ws = base.create_workspace(base.scene, state, inputs, .25, settings, context)
    ws.predict()
    ws.position_sweep()
    ws.prepare_velocity()
    ws.velocity_sweep()
    return ws


def test_all_request_validation_precedes_any_native_mutation_and_poison():
    _, state, inputs, plain, settings, context = particle_fixture(domains=("a", "b"))
    events = []
    base = observed_program(plain, events)
    ws = prepare(base, state, inputs, settings, context)
    try:
        with pytest.raises(ValueError):
            ws.apply_point_velocity_impulses(terminal_requests("a") + (
                PointVelocityImpulses("b", np.array([99]), np.zeros((1, 3))),))
        assert not any(name == "apply_velocity_impulses" for _, name in events)
        with pytest.raises(RuntimeError, match="poisoned"):
            ws.result()
    finally:
        ws.close()
    assert sum(name == "close" for _, name in events) == 2


def test_one_terminal_batch_per_domain_and_repeated_call_rejected():
    _, state, inputs, plain, settings, context = particle_fixture(domains=("a", "b"))
    events = []
    base = observed_program(plain, events)
    ws = prepare(base, state, inputs, settings, context)
    try:
        ws.apply_point_velocity_impulses(terminal_requests("a") + terminal_requests("b"))
        for name in ("a", "b"):
            assert events.count((name, "apply_velocity_impulses")) == 1
            np.testing.assert_array_equal(ws.result().domains[name].velocity[:, 0], [11 / 16, 0.])
        with pytest.raises(RuntimeError, match="once"):
            ws.apply_point_velocity_impulses(())
        with pytest.raises(RuntimeError, match="poisoned"):
            ws.result()
    finally:
        ws.close()
        ws.close()
    for name in ("a", "b"):
        assert events.count((name, "close")) == 1


def test_partial_construction_closes_preceding_participant():
    scene, state, inputs, plain, settings, context = particle_fixture(domains=("a", "b"))
    events = []
    base = observed_program(plain, events, failed_domain="b")
    with pytest.raises(SolverFailure, match="construction"):
        base.create_workspace(scene, state, inputs, .25, settings, context)
    assert events == [("a", "create"), ("a", "close")]


@pytest.mark.parametrize("phase", ["predict", "position_sweep", "prepare_velocity", "apply_velocity_impulses", "diagnostics", "close"])
def test_late_second_substep_failure_rolls_back_and_retry_is_fresh(phase):
    scene, state, inputs, plain, settings, context = particle_fixture(history=True)
    settings = replace(settings, substeps=2)
    events, count = [], {"fired": False}
    def hook(operation):
        if operation == phase and events.count(("cloth", "create")) == 2 and not count["fired"]:
            count["fired"] = True
            raise SolverFailure("injected late public operation")
    base = observed_program(plain, events, {"cloth": hook})
    wrapper = TerminalImpulseProgram(base, terminal_requests(), registry=base.registry)
    initial = fingerprint((state, inputs))
    with SceneStepper(scene, wrapper, settings, context, base.registry) as runtime:
        failed = runtime.step(state, inputs, .5)
    assert not failed.report.accepted and failed.report.completed_substeps == 1
    assert fingerprint(failed.state) == fingerprint(state)
    assert fingerprint((state, inputs)) == initial
    assert events.count(("cloth", "create")) == events.count(("cloth", "close"))
    base.participants["cloth"].hook = None
    with SceneStepper(scene, wrapper, settings, context, base.registry) as runtime:
        retry = runtime.step(state, inputs, .5)
    fresh_wrapper = TerminalImpulseProgram(plain, terminal_requests(), registry=plain.registry)
    with SceneStepper(scene, fresh_wrapper, settings, context, plain.registry) as runtime:
        fresh = runtime.step(state, inputs, .5)
    assert retry.report.accepted and fresh.report.accepted
    assert fingerprint(retry.state) == fingerprint(fresh.state)


@pytest.mark.parametrize("change", ["role", "backend", "history", "model"])
def test_mismatched_factory_contract_rejects_before_workspace(change):
    scene, _, _, plain, _, _ = particle_fixture(domains=("a", "b"))
    factory = ObservedFactory(plain.participants["b"], [], "b")
    if change == "role":
        factory.point_role = "articulated"
    elif change == "backend":
        factory.capabilities = replace(factory.capabilities, backend="other")
    elif change == "history":
        factory.capabilities = replace(factory.capabilities, history="explicit")
    else:
        factory.model = ParticleModel(np.ones(2), np.zeros(3))
    with pytest.raises(ValueError):
        PointParticipantProgram(scene, {"a": plain.participants["a"], "b": factory})


def test_binding_mutation_and_history_schema_are_not_silently_rebound():
    scene, state, inputs, plain, settings, context = particle_fixture(history=True)
    base = observed_program(plain, [])
    base.participants["cloth"].capabilities = replace(base.participants["cloth"].capabilities, features=frozenset())
    with pytest.raises(ValueError, match="changed"):
        base.create_workspace(scene, state, inputs, .25, settings, context)
    with pytest.raises(ValueError, match="duplicate"):
        PointParticipantProgram(scene, plain.participants, history_fields=plain.history_fields * 2)
    with pytest.raises(ValueError, match="history"):
        plain.create_workspace(scene, SceneState(state.domains), inputs, .25, settings, context)
