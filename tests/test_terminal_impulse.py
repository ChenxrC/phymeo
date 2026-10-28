"""Given terminal impulses exercise live native stages, never a contact solver."""

import numpy as np
import pytest

import phymeo as ph


def api():
    assert hasattr(ph, "PointVelocityImpulses"), "missing owned terminal request API"
    return ph.PointVelocityImpulses


def test_request_api_is_owned_and_signature_stable():
    request_type = api()
    indices = np.array([0, 0])
    impulses = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    local = np.zeros((2, 3))
    request = request_type("nodes", indices, impulses, local)
    from phymeo.core.scene import fingerprint

    signature = fingerprint(request)
    indices[:] = 5
    impulses[:] = 8
    local[:] = 4
    assert fingerprint(request) == signature
    assert not request.indices.flags.writeable
    assert not request.impulses.flags.writeable
    assert not request.local_points.flags.writeable
    np.testing.assert_array_equal(request.indices, [0, 0])
    assert request_type("nodes", np.empty(0, dtype=int), np.empty((0, 3))).impulses.shape == (0, 3)


@pytest.mark.parametrize(
    "indices,impulses,local",
    [
        (np.array([True]), np.zeros((1, 3)), None),
        (np.array([0.0]), np.zeros((1, 3)), None),
        (np.array([-1]), np.zeros((1, 3)), None),
        (np.array([0]), np.zeros((1, 3), dtype=np.float32), None),
        (np.array([0]), np.zeros((1, 2)), None),
        (np.array([0]), np.array([[0.0, np.nan, 0.0]]), None),
        (np.array([0]), np.array([[0.0, np.inf, 0.0]]), None),
        (np.array([0]), np.zeros((1, 3)), np.zeros((2, 3))),
        (np.array([0]), np.zeros((1, 3)), np.ones((1, 3), dtype=int)),
    ],
)
def test_request_rejects_malformed_values(indices, impulses, local):
    request_type = api()
    with pytest.raises((TypeError, ValueError)):
        request_type("nodes", indices, impulses, local)


def fixture(mode="numpy", chain=False):
    model = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 1.0, 0.0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    n = 2 if chain else 1
    particles = ph.ParticleModel(np.ones(n), np.zeros(3))
    material = (
        ph.DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([1 / 16]))
        if chain
        else None
    )
    state = ph.SceneState(
        {
            "robot": ph.RobotState(np.zeros(1), np.zeros(1)),
            "nodes": ph.ParticleState(
                np.array([[0.0, 1 / 8, 0.0], [0.0, 9 / 8, 0.0]])[:n],
                np.tile([0.0, -1.0, 0.0], (n, 1)),
            ),
        }
    )
    inputs = ph.SceneInputs({"robot": ph.RobotInputs.zeros(1), "nodes": ph.StepInputs.zeros(n)})
    scene = ph.SceneModel(
        (
            ph.DomainBinding("robot", "robot", model),
            ph.DomainBinding("nodes", "particles", particles),
        )
    )
    if mode == "numpy":
        participants = {
            "robot": ph.RobotParticipant(model),
            "nodes": ph.ClothParticipant(particles, material, record_material_path=True),
        }
        program_type = ph.ArticulatedCouplingProgram
    else:
        participants = {
            "robot": ph.TyxRobotParticipant(model, mode),
            "nodes": ph.TyxClothParticipant(particles, material, mode, record_material_path=True),
        }
        program_type = ph.TyxArticulatedCouplingProgram
    attachments = ph.RobotSphereAttachments(np.empty(0, dtype=int), np.empty((0, 3)), np.empty(0))
    base = program_type(scene, participants, attachments, ())
    settings = ph.StepSettings(substeps=1, tolerance=1e-8 if mode == "numpy" else 2e-5)
    context = ph.BackendContext(base.capabilities.backend)
    return scene, state, inputs, base, settings, context, material


def requests(eta=1 / 3):
    api()
    return (
        ph.PointVelocityImpulses("nodes", np.array([0]), np.array([[0.0, eta, 0.0]])),
        ph.PointVelocityImpulses(
            "robot", np.array([0]), np.array([[0.0, -eta, 0.0]]), np.zeros((1, 3))
        ),
    )


def run_case(mode="numpy", chain=False):
    assert hasattr(ph, "TerminalImpulseProgram"), "missing live terminal program API"
    scene, state, inputs, base, settings, context, material = fixture(mode, chain)
    assert "terminal_point_impulses" in base.capabilities.features
    wrapper = ph.TerminalImpulseProgram(base, requests(), registry=base.registry)
    p = 3 / 7 if chain else 1 / 3
    loads = np.zeros_like(state.domains["nodes"].position)
    loads[0, 1] = p
    with ph.SceneLoadReplay(
        scene,
        wrapper,
        state,
        inputs,
        0.25,
        channels={"nodes": ("force",), "robot": ("effort",)},
        settings=settings,
        context=context,
        registry=base.registry,
    ) as replay:
        trial = replay.evaluate({"nodes": {"force": loads}, "robot": {"effort": np.array([-p])}})
    assert trial.result.report.accepted, trial.result.report.reason
    return scene, state, inputs, base, settings, context, material, trial


@pytest.mark.parametrize("chain", [False, True])
def test_live_native_terminal_given_impulses(chain):
    *_, trial = run_case(chain=chain)
    evidence = trial.result.report.substeps[0].diagnostics["terminal_impulse"]
    pre = evidence["pre_impulse_state"]
    post = trial.result.state
    atol = 2e-13
    np.testing.assert_allclose(
        pre.domains["nodes"].velocity[:, 1],
        [-5 / 7, -6 / 7] if chain else [-2 / 3],
        atol=atol,
        rtol=0,
    )
    np.testing.assert_allclose(
        pre.domains["robot"].v, [-3 / 14 if chain else -1 / 6], atol=atol, rtol=0
    )
    np.testing.assert_allclose(
        post.domains["nodes"].velocity[:, 1],
        [-8 / 21, -6 / 7] if chain else [-1 / 3],
        atol=atol,
        rtol=0,
    )
    np.testing.assert_allclose(
        post.domains["robot"].v, [-8 / 21 if chain else -1 / 3], atol=atol, rtol=0
    )
    for domain, field in (("nodes", "position"), ("robot", "q")):
        assert (
            getattr(pre.domains[domain], field).tobytes()
            == getattr(post.domains[domain], field).tobytes()
        )
    assert evidence["applications"] == 1
    assert evidence["pre_position_iterations"] >= 1


def test_native_terminal_lifecycle_and_real_port_validation():
    scene, state, inputs, base, settings, context, _ = fixture()
    ws = base.create_workspace(scene, state, inputs, 0.25, settings, context)
    assert callable(getattr(ws, "apply_point_velocity_impulses", None)), "missing native forwarding"
    with pytest.raises(RuntimeError, match="prepare"):
        ws.apply_point_velocity_impulses(requests())
    ws.predict()
    ws.position_sweep()
    ws.prepare_velocity()
    with pytest.raises(ValueError):
        ws.apply_point_velocity_impulses(
            (ph.PointVelocityImpulses("unknown", np.array([0]), np.zeros((1, 3))),)
        )
    with pytest.raises(ValueError):
        ws.apply_point_velocity_impulses(
            (ph.PointVelocityImpulses("nodes", np.array([0]), np.zeros((1, 3)), np.ones((1, 3))),)
        )
    ws.apply_point_velocity_impulses(requests())
    ws.close()
    with pytest.raises(RuntimeError, match="closed"):
        ws.apply_point_velocity_impulses(requests())


def test_wrapper_applies_only_after_base_converges_once_and_never_after_close():
    assert hasattr(ph, "TerminalImpulseProgram"), "missing live terminal program API"
    scene, state, inputs, base, settings, context, _ = fixture()
    wrapper = ph.TerminalImpulseProgram(base, requests(), registry=base.registry)
    ws = wrapper.create_workspace(scene, state, inputs, 0.25, settings, context)
    ws.predict()
    ws.position_sweep()
    ws.prepare_velocity()
    ws.velocity_sweep()
    ws.velocity_metrics()
    before = ws.result().domains["nodes"].velocity.copy()
    for _ in range(3):
        ws.velocity_sweep()
        ws.velocity_metrics()
    np.testing.assert_array_equal(ws.result().domains["nodes"].velocity, before)
    assert ws.diagnostics()["terminal_impulse"]["applications"] == 1
    ws.close()
    with pytest.raises(RuntimeError, match="closed"):
        ws.velocity_metrics()


def ledger_fixture(chain=False):
    assert hasattr(ph, "TerminalImpulseLedger"), "missing independent terminal ledger API"
    scene, _, _, base, settings, context, _, trial = run_case(chain=chain)
    pre = trial.result.report.substeps[0].diagnostics["terminal_impulse"]["pre_impulse_state"]
    ledger = ph.TerminalImpulseLedger(
        scene, pre, requests(), settings, context, registry=base.registry
    )
    return scene, pre, trial.result.state, ledger, settings, context, base


@pytest.mark.parametrize("chain", [False, True])
def test_ledger_certifies_exact_velocity_equations_and_owns_report(chain):
    _, pre, post, ledger, _, _, _ = ledger_fixture(chain)
    report = ledger.require(post)
    assert report["max_residual"] < 1e-12
    assert report["configuration_preserved"]
    assert report["compute_precision"] == "float64"
    assert set(report["by_domain"]) == {"nodes", "robot"}
    with pytest.raises(TypeError):
        report["max_residual"] = 4
    assert not pre.domains["nodes"].velocity.flags.writeable


@pytest.mark.parametrize(
    "change", ["position", "q", "double_node", "missing_robot", "unrequested_node"]
)
def test_ledger_rejects_corrupted_response(change):
    scene, pre, post, ledger, _, context, base = ledger_fixture(True)
    altered = base.registry.clone_state(scene, post, context)
    if change == "position":
        altered.domains["nodes"].position[0, 0] = np.nextafter(0.0, 1.0)
    elif change == "q":
        altered.domains["robot"].q[0] = np.nextafter(altered.domains["robot"].q[0], np.inf)
    elif change == "double_node":
        altered.domains["nodes"].velocity[0, 1] += 1 / 3
    elif change == "missing_robot":
        altered.domains["robot"].v[:] = pre.domains["robot"].v
    else:
        altered.domains["nodes"].velocity[1, 0] += 0.01
    with pytest.raises(ph.SolverFailure):
        ledger.require(altered)


def test_ledger_empty_requests_requires_every_domain_and_history_unchanged():
    assert hasattr(ph, "TerminalImpulseLedger"), "missing independent terminal ledger API"
    scene, state, _, base, settings, context, _ = fixture()
    schema = (ph.HistoryField("base/count", (1,), np.int64),)
    state = ph.SceneState(state.domains, {"base/count": np.array([3], dtype=np.int64)})
    ledger = ph.TerminalImpulseLedger(
        scene, state, (), settings, registry=base.registry, history_fields=schema
    )
    assert ledger.require(state)["max_residual"] == 0
    altered = base.registry.clone_state(scene, state, context, schema)
    altered.history["base/count"][0] += 1
    with pytest.raises(ph.SolverFailure, match="history"):
        ledger.require(altered)
    altered = base.registry.clone_state(scene, state, context, schema)
    altered.domains["robot"].v[0] = 0.1
    with pytest.raises(ph.SolverFailure, match="unrequested"):
        ledger.require(altered)


def test_ledger_preserves_small_impulse_in_large_cancellation_and_speed_scale():
    assert hasattr(ph, "TerminalImpulseLedger"), "missing independent terminal ledger API"
    scene, state, _, base, settings, context, _ = fixture()
    req = (
        ph.PointVelocityImpulses(
            "nodes",
            np.array([0, 0, 0]),
            np.array([[0.0, 2.0**60, 0.0], [0.0, 1.0, 0.0], [0.0, -(2.0**60), 0.0]]),
        ),
    )
    ledger = ph.TerminalImpulseLedger(scene, state, req, settings, registry=base.registry)
    with pytest.raises(ph.SolverFailure, match="velocity"):
        ledger.require(state)
    correct = base.registry.clone_state(scene, state, context)
    correct.domains["nodes"].velocity[0, 1] += 1
    assert ledger.require(correct)["max_residual"] == 0
    state.domains["nodes"].velocity[:] = 77
    assert ledger.require(correct)["max_residual"] == 0


def test_ledger_rejects_unknown_precision_and_unsupported_requested_models(monkeypatch):
    from dataclasses import replace

    assert hasattr(ph, "TerminalImpulseLedger"), "missing independent terminal ledger API"
    scene, state, _, base, settings, _, _ = fixture()
    with pytest.raises(ValueError, match="context"):
        ph.TerminalImpulseLedger(
            scene, state, requests(), settings, ph.BackendContext("unknown"), registry=base.registry
        )
    model = replace(scene.domains[0].model, joint_types=("revolute",))
    unsupported = ph.SceneModel((ph.DomainBinding("robot", "robot", model), scene.domains[1]))
    with pytest.raises(ValueError, match="prismatic"):
        ph.TerminalImpulseLedger(unsupported, state, requests(), settings, registry=base.registry)
    import phymeo.validation.terminal_impulse as module

    monkeypatch.setattr(module, "mass_matrix", lambda *args: -np.ones((1, 1)))
    with pytest.raises(ph.SolverFailure, match="positive definite"):
        ph.TerminalImpulseLedger(scene, state, requests(), settings, registry=base.registry)


class ObservedProgram:
    """Real native stages plus explicit history and targeted fault injection."""

    def __init__(self, base, *, failure=None, stalled=False, missing=False):
        from dataclasses import replace

        self.base = base
        self.history_fields = (ph.HistoryField("base/count", (1,), np.int64),)
        self.capabilities = replace(base.capabilities, history="explicit")
        self.failure, self.stalled, self.missing = failure, stalled, missing
        self.workspaces = []
        self.close_calls = 0

    def problem_signature(self):
        return {
            "base": self.base.problem_signature(),
            "history": tuple((f.name, f.shape, f.dtype.str) for f in self.history_fields),
        }

    def create_workspace(self, *args):
        ws = ObservedWorkspace(self, self.base.create_workspace(*args), len(self.workspaces) + 1)
        self.workspaces.append(ws)
        if self.missing:
            ws.apply_point_velocity_impulses = None
        return ws

    def close(self):
        self.close_calls += 1
        self.base.close()


class ObservedWorkspace:
    def __init__(self, owner, native, index):
        self.owner, self.native, self.index = owner, native, index
        self.counts = {
            key: 0 for key in ("predict", "position", "prepare", "velocity", "terminal", "close")
        }

    def __getattr__(self, key):
        return getattr(self.native, key)

    def predict(self):
        self.counts["predict"] += 1
        self.native.predict()
        self.native.history["base/count"] += 1

    def position_sweep(self):
        self.counts["position"] += 1
        self.native.position_sweep()

    def prepare_velocity(self):
        self.counts["prepare"] += 1
        self.native.prepare_velocity()

    def velocity_sweep(self):
        self.counts["velocity"] += 1
        self.native.velocity_sweep()

    def velocity_metrics(self):
        self.native.velocity_metrics()
        return (
            ph.StageMetrics({"stalled": 1.0})
            if self.owner.stalled
            else self.native.velocity_metrics()
        )

    def apply_point_velocity_impulses(self, values):
        self.counts["terminal"] += 1
        if self.owner.failure == "partial" and self.index == 2:
            participant = self.native.participants["robot"]
            prior = self.native.participants["nodes"].velocity().copy()

            def fail_after_node(*args):
                current = self.native.participants["nodes"].velocity()
                assert not np.array_equal(prior, current), "first domain must already be applied"
                raise ph.SolverFailure("injected second-domain terminal failure")

            participant.apply_velocity_impulses = fail_after_node
        self.native.apply_point_velocity_impulses(values)
        if self.owner.failure == "small_velocity":
            self.native.participants["nodes"].native.candidate.velocity[0, 1] += 5e-9
        if self.owner.failure == "history":
            self.native.history["base/count"] += 1
        if self.owner.failure == "terminal" and self.index == 2:
            raise ph.SolverFailure("injected late terminal failure")

    def close(self):
        self.counts["close"] += 1
        self.native.close()
        if self.owner.failure == "close" and self.index == 2:
            raise ph.SolverFailure("injected late close failure")


def test_terminal_wrapper_independently_rejects_history_changed_by_native_operation():
    scene, state, inputs, base, settings, context, _ = fixture()
    observed = ObservedProgram(base, failure="history")
    state = ph.SceneState(state.domains, {"base/count": np.array([0], dtype=np.int64)})
    wrapper = ph.TerminalImpulseProgram(observed, requests(), registry=base.registry)
    with ph.SceneStepper(scene, wrapper, settings, context, base.registry) as runtime:
        result = runtime.step(state, inputs, 0.25)
    assert not result.report.accepted, "terminal wrapper must certify native phase history"
    assert "history" in result.report.reason


@pytest.mark.parametrize("failure", [None, "terminal", "partial", "close"])
def test_explicit_history_and_late_failure_rollback_retry_matches_fresh(failure):
    from dataclasses import replace

    from phymeo.core.scene import fingerprint

    scene, state, inputs, base, settings, context, _ = fixture()
    state = ph.SceneState(state.domains, {"base/count": np.array([0], dtype=np.int64)})
    initial_signature = fingerprint((state, inputs))
    observed = ObservedProgram(base, failure=failure)
    wrapper = ph.TerminalImpulseProgram(observed, requests(), registry=base.registry)
    settings = replace(settings, substeps=2, record_substeps=True)
    with ph.SceneStepper(scene, wrapper, settings, context, base.registry) as runtime:
        result = runtime.step(state, inputs, 0.5)
        assert result.report.accepted == (failure is None)
        if failure:
            assert result.report.completed_substeps == 1
            assert fingerprint(result.state) == fingerprint(state)
            assert result.substep_states == ()
            result = runtime.step(state, inputs, 0.5)
            assert result.report.accepted, result.report.reason
    assert fingerprint((state, inputs)) == initial_signature
    assert observed.close_calls == 0
    assert result.state.history["base/count"][0] == 2
    assert [s.history["base/count"][0] for s in result.substep_states] == [1, 2]
    for ws in observed.workspaces:
        assert ws.counts == dict(predict=1, position=1, prepare=1, velocity=1, terminal=1, close=1)
    fresh_observed = ObservedProgram(base)
    fresh_wrapper = ph.TerminalImpulseProgram(fresh_observed, requests(), registry=base.registry)
    with ph.SceneStepper(scene, fresh_wrapper, settings, context, base.registry) as runtime:
        fresh = runtime.step(state, inputs, 0.5)
    assert fingerprint(result.state) == fingerprint(fresh.state)
    observed.close()
    assert observed.close_calls == 1


def test_nonconverged_base_cannot_be_masked_and_missing_operation_cleans_up():
    from dataclasses import replace

    scene, state, inputs, base, settings, context, _ = fixture()
    state = ph.SceneState(state.domains, {"base/count": np.array([0], dtype=np.int64)})
    stalled = ObservedProgram(base, stalled=True)
    wrapper = ph.TerminalImpulseProgram(stalled, requests(), registry=base.registry)
    with ph.SceneStepper(
        scene, wrapper, replace(settings, velocity_iterations=2), context, base.registry
    ) as runtime:
        result = runtime.step(state, inputs, 0.25)
    assert not result.report.accepted and "velocity iteration budget" in result.report.reason
    assert stalled.workspaces[0].counts["terminal"] == 0
    assert stalled.workspaces[0].counts["velocity"] == 2
    missing = ObservedProgram(base, missing=True)
    with pytest.raises(ValueError, match="callable"):
        ph.TerminalImpulseProgram(missing, (), registry=base.registry).create_workspace(
            scene, state, inputs, 0.25, settings, context
        )
    assert missing.workspaces[0].counts["close"] == 1


def test_empty_requests_preserve_native_values_and_iteration_counts():
    from phymeo.core.scene import fingerprint

    scene, state, inputs, base, settings, context, _ = fixture(chain=True)
    with ph.SceneStepper(scene, base, settings, context, base.registry) as runtime:
        original = runtime.step(state, inputs, 0.25)
    wrapped = ph.TerminalImpulseProgram(base, (), registry=base.registry)
    with ph.SceneStepper(scene, wrapped, settings, context, base.registry) as runtime:
        actual = runtime.step(state, inputs, 0.25)
    assert actual.report.accepted and original.report.accepted
    assert fingerprint(actual.state) == fingerprint(original.state)
    a, b = actual.report.substeps[0], original.report.substeps[0]
    assert (a.position_iterations, a.velocity_iterations, a.position, a.velocity) == (
        b.position_iterations,
        b.velocity_iterations,
        b.position,
        b.velocity,
    )
    assert fingerprint(a.diagnostics["participants"]) == fingerprint(b.diagnostics["participants"])


def test_owned_diagnostics_freezes_final_dataclass_arrays_without_borrowed_alias():
    from phymeo.runtime.program import own_diagnostics

    _, state, _, _, _, _, _ = fixture()
    copied = own_diagnostics({"pre": state})["pre"]
    for domain, fields in (("nodes", ("position", "velocity")), ("robot", ("q", "v"))):
        for field in fields:
            source = getattr(state.domains[domain], field)
            output = getattr(copied.domains[domain], field)
            assert source.flags.writeable and not output.flags.writeable
            assert not np.shares_memory(source, output)
            value = output.copy()
            source[:] = 42
            np.testing.assert_array_equal(output, value)


def test_ledger_audits_fp32_request_conversion_in_speed_units_even_when_equation_is_exact():
    scene, state, _, base, settings, _, _ = fixture()
    requested = ph.PointVelocityImpulses(
        "nodes", np.array([0]), np.array([[0.0, 2.0**24 + 1.0, 0.0]])
    )
    candidate = base.registry.clone_state(scene, state, ph.BackendContext())
    candidate.domains["nodes"].velocity[0, 1] = 2.0**24
    ledger = ph.TerminalImpulseLedger(
        scene,
        state,
        (requested,),
        settings,
        ph.BackendContext("tyx:interpret"),
        registry=base.registry,
    )
    with pytest.raises(ph.SolverFailure, match="input_mapping"):
        ledger.require(candidate)


def test_ledger_speed_scale_is_absolute_and_robot_mass_is_solved():
    from dataclasses import replace

    scene, state, _, base, settings, context, _ = fixture()
    settings = replace(settings, reference_speed=2.0, tolerance=0.01)
    candidate = base.registry.clone_state(scene, state, context)
    candidate.domains["nodes"].velocity[0, 1] += 1 / 3 + 0.01
    candidate.domains["robot"].v[0] -= 1 / 6 - 0.01
    ledger = ph.TerminalImpulseLedger(scene, state, requests(), settings, registry=base.registry)
    report = ledger.require(candidate)
    np.testing.assert_allclose(report["by_domain"]["nodes"]["velocity"], 0.005, rtol=0, atol=1e-15)
    np.testing.assert_allclose(report["by_domain"]["robot"]["velocity"], 0.005, rtol=0, atol=1e-15)
    candidate.domains["robot"].v[0] += 0.02
    with pytest.raises(ph.SolverFailure, match="robot velocity"):
        ledger.require(candidate)


def test_binding_validation_fixed_node_unknown_domain_bad_local_and_bounds():
    from dataclasses import replace

    scene, state, _, base, settings, _, _ = fixture()
    fixed = replace(scene.domains[1].model, inverse_mass=np.zeros(1))
    fixed_scene = ph.SceneModel((scene.domains[0], ph.DomainBinding("nodes", "particles", fixed)))
    for altered_scene, req, message in (
        (fixed_scene, requests(), "fully dynamic"),
        (scene, (ph.PointVelocityImpulses("nodes", np.array([1]), np.zeros((1, 3))),), "outside"),
        (scene, (ph.PointVelocityImpulses("unknown", np.array([0]), np.zeros((1, 3))),), "unknown"),
        (
            scene,
            (ph.PointVelocityImpulses("nodes", np.array([0]), np.zeros((1, 3)), np.ones((1, 3))),),
            "offset",
        ),
    ):
        with pytest.raises((ValueError, ph.SolverFailure), match=message):
            ph.TerminalImpulseLedger(altered_scene, state, req, settings, registry=base.registry)


def test_same_domain_batches_preserve_duplicate_requests_and_shared_tree_response():
    model = ph.RobotModel(
        np.array([-1, 0]),
        ("prismatic", "prismatic"),
        np.zeros((2, 3)),
        np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([2.0, 3.0]),
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.zeros(3),
    )
    scene = ph.SceneModel((ph.DomainBinding("robot", "robot", model),))
    state = ph.SceneState({"robot": ph.RobotState(np.zeros(2), np.zeros(2))})
    inputs = ph.SceneInputs({"robot": ph.RobotInputs.zeros(2)})
    empty = ph.RobotSphereAttachments(np.empty(0, dtype=int), np.empty((0, 3)), np.empty(0))
    base = ph.ArticulatedCouplingProgram(scene, {"robot": ph.RobotParticipant(model)}, empty, ())
    req = tuple(
        ph.PointVelocityImpulses(
            "robot", np.array([1]), np.array([[2.5, 3.0, 0.0]]), np.zeros((1, 3))
        )
        for _ in range(2)
    )
    wrapped = ph.TerminalImpulseProgram(base, req, registry=base.registry)
    settings = ph.StepSettings(substeps=1, tolerance=1e-8)
    with ph.SceneStepper(scene, wrapped, settings, registry=base.registry) as runtime:
        result = runtime.step(state, inputs, 0.25)
    assert result.report.accepted, result.report.reason
    np.testing.assert_array_equal(result.state.domains["robot"].v, [1.0, 2.0])
    evidence = result.report.substeps[0].diagnostics["terminal_impulse"]
    assert len(evidence["requests"]) == 2 and len(evidence["batches"]) == 1
    np.testing.assert_array_equal(evidence["batches"][0].indices, [1, 1])
    assert evidence["ledger"]["max_residual"] == 0


def test_numpy_original_material_equations_accept_pre_and_reject_terminal_post():
    from fractions import Fraction as F

    scene, initial, _, _, settings, context, material, trial = run_case(chain=True)
    evidence = trial.result.report.substeps[0].diagnostics["terminal_impulse"]
    pre = evidence["pre_impulse_state"]
    original = ph.StepInputs.zeros(2)
    book = ph.MaterialPathLedger(
        scene.domains[1].model,
        material,
        initial.domains["nodes"],
        original,
        0.25,
        settings,
        context,
    )
    expected = np.full((2, 3), F(0), dtype=object)
    expected[0, 1] = F(3, 7)
    diag = evidence["pre_impulse_diagnostics"]["participants"]["nodes"]
    book.require(
        pre.domains["nodes"],
        trial.effective_inputs.domains["nodes"],
        expected,
        diag,
        sweeps=evidence["pre_position_iterations"],
    )
    with pytest.raises(ph.SolverFailure, match="velocity"):
        book.require(
            trial.result.state.domains["nodes"],
            trial.effective_inputs.domains["nodes"],
            expected,
            diag,
            sweeps=evidence["pre_position_iterations"],
        )
    from phymeo.core.scene import fingerprint

    assert fingerprint(diag["material_path"]) == fingerprint(
        trial.result.report.substeps[0].diagnostics["participants"]["nodes"]["material_path"]
    )


def test_terminal_program_owns_requests_signature_and_removes_unforwarded_capability():
    from phymeo.core.scene import fingerprint

    base = fixture()[3]
    original = requests()
    wrapper = ph.TerminalImpulseProgram(base, original, registry=base.registry)
    signature = fingerprint(wrapper.problem_signature())
    original[0].impulses.flags.writeable = True
    original[0].impulses[:] = 77
    assert fingerprint(wrapper.problem_signature()) == signature
    assert "terminal_point_impulses" not in wrapper.capabilities.features
    assert "terminal_point_impulse_pass" in wrapper.capabilities.features


def test_native_contact_program_does_not_claim_terminal_capability():
    scene, _, _, base, _, _, _ = fixture()
    attachments = ph.RobotSphereAttachments(np.array([0]), np.zeros((1, 3)), np.ones(1))
    contact = ph.NodeRobotContact("nodes", np.array([0]), np.array([0]))
    constrained = ph.ArticulatedCouplingProgram(scene, base.participants, attachments, (contact,))
    assert "terminal_point_impulses" not in constrained.capabilities.features
    with pytest.raises(ValueError, match="capability"):
        ph.TerminalImpulseProgram(constrained, requests(), registry=base.registry)


def test_terminal_evidence_retains_both_original_base_metrics():
    *_, trial = run_case()
    report = trial.result.report.substeps[0]
    evidence = report.diagnostics["terminal_impulse"]
    assert evidence.get("pre_position_metrics") == report.position
    original = evidence["pre_velocity_metrics"]
    for key, value in original.residuals.items():
        assert report.velocity.residuals[f"base/{key}"] == value
    assert report.velocity.residuals["terminal_impulse"] == evidence["ledger"]["max_residual"]


def test_device_ledger_output_accepts_exact_subnormal_fp32_promotions():
    scene, state, _, base, settings, _, _ = fixture()
    state.domains["nodes"].velocity[:] = float(np.nextafter(np.float32(0), np.float32(1)))
    req = (ph.PointVelocityImpulses("nodes", np.array([0]), np.zeros((1, 3))),)
    ledger = ph.TerminalImpulseLedger(
        scene, state, req, settings, ph.BackendContext("tyx:interpret"), registry=base.registry
    )
    assert ledger.require(state)["max_residual"] == 0


def test_final_stage_metrics_include_tolerable_terminal_equation_error():
    scene, state, inputs, base, settings, context, _ = fixture()
    state = ph.SceneState(state.domains, {"base/count": np.array([0], dtype=np.int64)})
    observed = ObservedProgram(base, failure="small_velocity")
    wrapper = ph.TerminalImpulseProgram(observed, requests(), registry=base.registry)
    with ph.SceneStepper(scene, wrapper, settings, context, base.registry) as runtime:
        result = runtime.step(state, inputs, 0.25)
    assert result.report.accepted, result.report.reason
    report = result.report.substeps[0]
    evidence = report.diagnostics["terminal_impulse"]
    terminal_error = evidence["ledger"]["max_residual"]
    assert 0 < terminal_error < settings.tolerance
    assert result.report.max_residual >= terminal_error
    assert report.velocity.geometric_error == evidence["pre_velocity_metrics"].geometric_error
    assert report.velocity.penetration == evidence["pre_velocity_metrics"].penetration
