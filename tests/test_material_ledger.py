"""Independent equation certification of native distance material observations."""

from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from test_material_path import material_case

import phymeo as ph


def collect(mode="numpy", kind="shared", *, setup=None, force=None, h=0.1):
    model, initial, constraints = material_case(kind)
    if kind in ("fixed", "fixed_compliant"):
        if kind == "fixed_compliant":
            model = replace(model, inverse_mass=np.zeros(3))
        initial.velocity[model.inverse_mass == 0] = 0
    if setup is not None:
        model, initial, constraints = setup
    inputs = ph.StepInputs.zeros(model.particle_count) if force is None else ph.StepInputs(force)
    settings = ph.StepSettings(substeps=1, tolerance=1e-8 if mode == "numpy" else 2e-5)
    factory = (
        ph.ClothParticipant(model, constraints, record_material_path=True)
        if mode == "numpy"
        else ph.TyxClothParticipant(model, constraints, mode, record_material_path=True)
    )
    context = ph.BackendContext(factory.capabilities.backend)
    ws = factory.create_workspace("cloth", initial, inputs, h, settings, context)
    try:
        ws.predict()
        for count in range(1, settings.position_iterations + 1):
            ws.position_sweep()
            if ws.position_metrics().max_residual <= settings.tolerance:
                break
        else:
            raise AssertionError("fixture convergence")
        ws.prepare_velocity()
        final, diag = ws.result(), ws.diagnostics()
    finally:
        ws.close()
    return dict(
        model=model,
        initial=initial,
        constraints=constraints,
        inputs=inputs,
        settings=settings,
        context=context,
        h=h,
        final=final,
        diag=diag,
        sweeps=count,
    )


def ledger(case, original=None):
    return ph.MaterialPathLedger(
        case["model"],
        case["constraints"],
        case["initial"],
        case["inputs"] if original is None else original,
        case["h"],
        case["settings"],
        case["context"],
    )


def require(case, *, book=None, diag=None, final=None, expected=None):
    if expected is None:
        expected = np.zeros_like(case["initial"].position)
    return (ledger(case) if book is None else book).require(
        case["final"] if final is None else final,
        case["inputs"],
        expected,
        case["diag"] if diag is None else diag,
        sweeps=case["sweeps"],
    )


def mutable(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {k: mutable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(mutable(v) for v in value)
    return value


@pytest.mark.parametrize(
    "kind", ["shared", "fixed", "fixed_compliant", "coincident", "none", "empty"]
)
def test_native_material_ledger_accepts_independent_equations(kind):
    c = collect(kind=kind)
    report = require(c)
    assert report["scope"] == "distance-material-equations-v1"
    assert report["max_residual"] <= c["settings"].tolerance
    assert not report["material_impulses"].flags.writeable
    np.testing.assert_allclose(report["material_impulses"].sum(axis=0), 0, atol=1e-14)
    if kind == "shared":
        assert report["energy"]["kinetic_after"] > report["energy"]["kinetic_before"]
        assert report["energy"]["change_minus_work"][1] < 0


@pytest.mark.parametrize(
    "field",
    [
        "before",
        "after",
        "direction",
        "delta",
        "lambda",
        "alpha",
        "prediction",
        "final_lambda",
        "columns",
        "missing",
    ],
)
def test_corrupt_path_cannot_be_accepted(field):
    c = collect()
    book = ledger(c)
    d = mutable(c["diag"])
    path = d["material_path"]
    if field in ("before", "after", "direction", "delta", "lambda"):
        column = {"before": 0, "after": 6, "direction": 14, "delta": 17, "lambda": 13}[field]
        path["frames"][0][0, column] += 0.01
    elif field == "alpha":
        path["alpha"][0] += 0.1
    elif field == "prediction":
        path["predicted_velocity"][0, 0] += 0.01
    elif field == "final_lambda":
        d["distance_multipliers"][0] += 0.01
    elif field == "columns":
        path["columns"] = tuple(reversed(path["columns"]))
    else:
        del d["material_path"]
    with pytest.raises(ph.SolverFailure):
        require(c, book=book, diag=d)


@pytest.mark.parametrize("field", ["position", "velocity"])
def test_hidden_post_material_load_is_rejected(field):
    c = collect()
    book = ledger(c)
    final = c["final"].copy()
    getattr(final, field)[0, 0] += 0.001
    with pytest.raises(ph.SolverFailure):
        require(c, book=book, final=final)


def test_ledger_snapshots_original_inputs_and_owns_reports():
    c = collect()
    book = ledger(c)
    report = require(c, book=book)
    c["initial"].position[:] = 999
    repeated = require(c, book=book)
    np.testing.assert_array_equal(report["material_impulses"], repeated["material_impulses"])
    assert not np.shares_memory(report["material_impulses"], repeated["material_impulses"])


def test_original_impulse_request_not_rounded_input_difference():
    c = collect(force=np.full((3, 3), 0.2))
    original = ph.StepInputs.zeros(3)
    book = ledger(c, original)
    impulse = np.full((3, 3), Fraction(1, 50), dtype=object)
    report = require(c, book=book, expected=impulse)
    assert report["max_residual"] <= c["settings"].tolerance
    with pytest.raises(ph.SolverFailure):
        require(c, book=book, expected=2 * impulse)


def test_public_ledger_configuration_rejects_unknown_context_and_moving_support():
    c = collect()
    with pytest.raises(ValueError):
        ph.MaterialPathLedger(
            c["model"],
            c["constraints"],
            c["initial"],
            c["inputs"],
            0.1,
            c["settings"],
            ph.BackendContext("unknown"),
        )
    c = collect(kind="fixed")
    c["initial"].velocity[0, 0] = 1
    with pytest.raises(ValueError):
        ledger(c)


@pytest.mark.parametrize("bad", [True, 0, 1.5, 1000])
def test_sweep_count_must_match_recorded_schedule(bad):
    c = collect()
    book = ledger(c)
    with pytest.raises(ph.SolverFailure):
        book.require(c["final"], c["inputs"], np.zeros((3, 3)), c["diag"], sweeps=bad)


def test_hard_incompatible_initial_shape_reports_energy_creation_without_claiming_conservation():
    setup = (
        ph.ParticleModel(np.ones(2), np.zeros(3)),
        ph.ParticleState(np.array([[0.0, 0, 0], [2.0, 0, 0]]), np.zeros((2, 3))),
        ph.DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.zeros(1)),
    )
    c = collect(setup=setup)
    r = require(c)
    np.testing.assert_allclose(r["material_impulses"], [[5, 0, 0], [-5, 0, 0]], atol=1e-14)
    np.testing.assert_allclose(r["energy"]["change_minus_work"], [25, 25], atol=1e-13)
    assert r["scope"] == "distance-material-equations-v1"


@pytest.mark.parametrize("h", [0.1, 0.05, 0.025])
def test_angular_loss_is_reported_as_material_discrete_contribution(h):
    setup = (
        ph.ParticleModel(np.ones(2), np.zeros(3)),
        ph.ParticleState(
            np.array([[-1.0, 0, 0], [1.0, 0, 0]]), np.array([[0.0, -1, 0], [0.0, 1, 0]])
        ),
        ph.DistanceConstraints(np.array([[0, 1]]), np.array([2.0]), np.zeros(1)),
    )
    c = collect(setup=setup, h=h)
    r = require(c)
    loss = 2 / np.sqrt(1 + h * h) - 2
    np.testing.assert_allclose(r["momentum"]["angular_change"], [0, 0, loss], atol=1e-14)
    np.testing.assert_allclose(r["momentum"]["material_angular_impulse"], [0, 0, loss], atol=1e-14)
    assert r["momentum"]["angular_change"][2] < 0


def test_fixed_compliant_path_cannot_hide_intermediate_equation_error():
    c = collect(kind="fixed_compliant")
    d = mutable(c["diag"])
    first = d["material_path"]["frames"][0]
    second = first.copy()
    second[:, 12] = first[:, 13]
    second[:, 17] = 0
    first[0, 17] += 0.1
    first[0, 13] += 0.1
    second[0, 12] = first[0, 13]
    second[0, 17] = -0.1
    d["material_path"]["frames"] = (first, second)
    c["sweeps"] = 2
    with pytest.raises(ph.SolverFailure, match="edge_equation"):
        require(c, diag=d)


def test_small_fixed_edge_errors_accumulate_without_iteration_dilution():
    c = collect(kind="fixed_compliant")
    d = mutable(c["diag"])
    first = d["material_path"]["frames"][0]
    repeated = first.copy()
    repeated[:, 12] = first[:, 13]
    repeated[:, 17] = 0
    denominator = c["constraints"].compliance[0] / c["h"] ** 2
    repeated[0, 17] = 0.3 * c["settings"].tolerance / denominator
    d["material_path"]["frames"] = (first,) + (repeated.copy(),) * 5
    c["sweeps"] = 6
    with pytest.raises(ph.SolverFailure, match="edge_equation|lambda_update"):
        require(c, diag=d)


def test_lost_request_beside_large_original_force_is_rejected():
    c = collect(kind="none")
    c["inputs"].force[:] = 1e16
    book = ledger(c)
    request = np.zeros((3, 3), object)
    request[0, 0] = Fraction(1)
    with pytest.raises(ph.SolverFailure, match="input"):
        require(c, book=book, expected=request)


def test_large_world_coordinate_does_not_enlarge_prediction_budget():
    setup = (
        ph.ParticleModel(np.ones(1), np.zeros(3)),
        ph.ParticleState(np.array([[1e8, 0, 0]]), np.array([[1e-7, 0, 0]])),
        None,
    )
    c = collect(setup=setup, h=0.01)
    with pytest.raises(ph.SolverFailure, match="prediction_position"):
        require(c)


def test_gravity_energy_and_declared_displacement_work_are_distinct():
    setup = (
        ph.ParticleModel(np.ones(1), np.array([0.0, -10, 0])),
        ph.ParticleState(np.zeros((1, 3)), np.zeros((1, 3))),
        None,
    )
    c = collect(setup=setup)
    r = require(c)
    np.testing.assert_allclose(r["energy"]["total_after"], [-0.5, -0.5], atol=1e-14)
    assert r["energy"]["endpoint_work"] == 0
    np.testing.assert_allclose(r["momentum"]["external_impulse"], [0, -1, 0], atol=1e-14)


def check_audited_transaction(mode, failure):
    from phymeo.runtime.program import own_diagnostics

    model, initial, constraints = material_case()
    factory = (
        ph.ClothParticipant(model, constraints, record_material_path=True)
        if mode == "numpy"
        else ph.TyxClothParticipant(model, constraints, mode, record_material_path=True)
    )
    scene = ph.SceneModel((ph.DomainBinding("cloth", "particles", model),))

    class Program:
        capabilities = replace(factory.capabilities, history="explicit")
        history_fields = (ph.HistoryField("count", (1,), np.int64),)
        fail = failure
        created = closed = 0

        def problem_signature(self):
            return factory.problem_signature()

        def create_workspace(self, scene, state, inputs, h, settings, context):
            native = factory.create_workspace(
                "cloth", state.domains["cloth"], inputs.domains["cloth"], h, settings, context
            )
            book = ph.MaterialPathLedger(
                model,
                constraints,
                state.domains["cloth"],
                inputs.domains["cloth"],
                h,
                settings,
                context,
            )
            count = state.history["count"][0]
            owner = self
            owner.created += 1

            class Workspace:
                sweeps = 0
                report = None

                def __getattr__(self, name):
                    return getattr(native, name)

                def position_sweep(self):
                    native.position_sweep()
                    self.sweeps += 1

                def result(self):
                    result = native.result()
                    diag = mutable(native.diagnostics())
                    if owner.fail == "path" and count == 1:
                        diag["material_path"]["frames"][-1][0, 6] += 0.01
                    self.report = book.require(
                        result, inputs.domains["cloth"], np.zeros((3, 3)), diag, sweeps=self.sweeps
                    )
                    return ph.SceneState(
                        {"cloth": result}, {"count": np.array([count + 1], np.int64)}
                    )

                def diagnostics(self):
                    return own_diagnostics(dict(native.diagnostics(), material_audit=self.report))

                def close(self):
                    native.close()
                    owner.closed += 1
                    if owner.fail == "close" and count == 1:
                        raise ph.SolverFailure("late audited close")

            return Workspace()

        def close(self):
            pass

    state = ph.SceneState({"cloth": initial}, {"count": np.zeros(1, np.int64)})
    inputs = ph.SceneInputs({"cloth": ph.StepInputs.zeros(3)})
    settings = ph.StepSettings(substeps=2, tolerance=1e-8 if mode == "numpy" else 2e-5)
    context = ph.BackendContext(factory.capabilities.backend)
    program = Program()
    with ph.SceneStepper(scene, program, settings, context) as runtime:
        failed = runtime.step(state, inputs, 0.2)
        assert not failed.report.accepted and len(failed.report.substeps) == 2
        np.testing.assert_array_equal(failed.state.domains["cloth"].position, initial.position)
        np.testing.assert_array_equal(failed.state.domains["cloth"].velocity, initial.velocity)
        np.testing.assert_array_equal(failed.state.history["count"], [0])
        assert program.created == program.closed == 2
        program.fail = None
        retry = runtime.step(state, inputs, 0.2)
    fresh = Program()
    fresh.fail = None
    with ph.SceneStepper(scene, fresh, settings, context) as runtime:
        expected = runtime.step(state, inputs, 0.2)
    assert retry.report.accepted and expected.report.accepted
    np.testing.assert_array_equal(
        retry.state.domains["cloth"].position, expected.state.domains["cloth"].position
    )
    np.testing.assert_array_equal(
        retry.state.domains["cloth"].velocity, expected.state.domains["cloth"].velocity
    )
    np.testing.assert_array_equal(retry.state.history["count"], [2])


@pytest.mark.parametrize("failure", ["path", "close"])
def test_audited_later_substep_failure_restores_history_and_retry(failure):
    check_audited_transaction("numpy", failure)


def test_load_replay_material_requests_are_audited_from_original_snapshot():
    from phymeo.solvers.cloth import ClothProgram

    model, state, constraints = material_case()
    program = ClothProgram(model, constraints, record_material_path=True)
    inputs = ph.StepInputs.zeros(3)
    settings = ph.StepSettings(substeps=1, tolerance=1e-8)
    book = ph.MaterialPathLedger(model, constraints, state, inputs, 0.1, settings)
    initial = ph.SceneState({"cloth": state})
    forces = ph.SceneInputs({"cloth": inputs})
    outputs = []
    with ph.SceneLoadReplay(
        program.scene,
        program,
        initial,
        forces,
        0.1,
        channels={"cloth": ("force",)},
        settings=settings,
    ) as replay:
        for magnitude in (0.02, -0.03, 0.02):
            requested = np.zeros((3, 3))
            requested[1, 1] = magnitude
            trial = replay.evaluate({"cloth": {"force": requested}})
            assert trial.result.report.accepted
            sub = trial.result.report.substeps[0]
            audit = book.require(
                trial.result.state.domains["cloth"],
                trial.effective_inputs.domains["cloth"],
                requested,
                sub.diagnostics,
                sweeps=sub.position_iterations,
            )
            assert audit["max_residual"] <= settings.tolerance
            outputs.append(trial.result.state.domains["cloth"].position)
        np.testing.assert_array_equal(replay.snapshot().domains["cloth"].position, state.position)
    np.testing.assert_array_equal(outputs[0], outputs[2])
    assert not np.array_equal(outputs[0], outputs[1])


def check_triangle_robot_requested_pair(mode):
    from phymeo.solvers._robot_replay_ledger import FrozenRobotLedger

    robot = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0, 0, 1]]),
        np.array([[0.0, 1, 0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    model = ph.ParticleModel(np.ones(3), np.zeros(3))
    nodes = ph.ParticleState(np.array([[1.0, 0, 0], [2.0, 0, 0], [1.0, 1, 0]]), np.zeros((3, 3)))
    constraints = ph.DistanceConstraints(
        np.array([[0, 1], [1, 2], [2, 0]]), np.array([1.0, np.sqrt(2), 1.0]), np.full(3, 0.001)
    )
    scene = ph.SceneModel(
        (ph.DomainBinding("robot", "robot", robot), ph.DomainBinding("nodes", "particles", model))
    )
    state = ph.SceneState({"robot": ph.RobotState(np.zeros(1), np.zeros(1)), "nodes": nodes})
    inputs = ph.SceneInputs({"robot": ph.RobotInputs.zeros(1), "nodes": ph.StepInputs.zeros(3)})
    if mode == "numpy":
        participants = {
            "robot": ph.RobotParticipant(robot),
            "nodes": ph.ClothParticipant(model, constraints, record_material_path=True),
        }
        factory = ph.ArticulatedCouplingProgram
    else:
        participants = {
            "robot": ph.TyxRobotParticipant(robot, mode),
            "nodes": ph.TyxClothParticipant(model, constraints, mode, record_material_path=True),
        }
        factory = ph.TyxArticulatedCouplingProgram
    base = factory(
        scene,
        participants,
        ph.RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0)),
        [],
    )
    settings = ph.StepSettings(substeps=1, tolerance=1e-8 if mode == "numpy" else 2e-5)
    context = ph.BackendContext(base.capabilities.backend)
    book = ph.MaterialPathLedger(
        model, constraints, nodes, inputs.domains["nodes"], 0.1, settings, context
    )
    robot_book = FrozenRobotLedger(
        robot, state.domains["robot"], inputs.domains["robot"], 0.1, settings, 1.0
    )
    requested = np.zeros((3, 3))
    requested[0, 1] = 0.2
    # Prescribed opposing inputs, deliberately no collision/contact solver here.
    with ph.SceneLoadReplay(
        scene,
        base,
        state,
        inputs,
        0.1,
        channels={"robot": ("effort",), "nodes": ("force",)},
        settings=settings,
        context=context,
        registry=base.registry,
    ) as replay:
        trial = replay.evaluate(
            {"robot": {"effort": np.array([-0.2])}, "nodes": {"force": requested}}
        )
    assert trial.result.report.accepted
    sub = trial.result.report.substeps[0]
    report = book.require(
        trial.result.state.domains["nodes"],
        trial.effective_inputs.domains["nodes"],
        requested,
        sub.diagnostics["participants"]["nodes"],
        sweeps=sub.position_iterations,
    )
    residual = robot_book.trial_audit(
        trial.result.state.domains["robot"],
        trial.effective_inputs.domains["robot"],
        [-Fraction(0.2)],
    )
    assert max(residual.values()) <= settings.tolerance
    py = (
        trial.result.state.domains["nodes"].velocity[:, 1].sum()
        + 2 * trial.result.state.domains["robot"].v[0]
    )
    assert abs(py) < 2e-6
    assert report["max_residual"] <= settings.tolerance


def test_triangle_and_actual_robot_pair_inputs_are_each_accounted_once():
    check_triangle_robot_requested_pair("numpy")


def test_parameter_and_original_force_snapshots_do_not_alias_callers():
    c = collect()
    book = ledger(c)
    c["model"].inverse_mass.flags.writeable = True
    c["model"].inverse_mass[:] = 10
    c["constraints"].rest_length.flags.writeable = True
    c["constraints"].rest_length[:] = 99
    c["inputs"].force[:] = 7
    report = book.require(
        c["final"], ph.StepInputs.zeros(3), np.zeros((3, 3)), c["diag"], sweeps=c["sweeps"]
    )
    assert report["max_residual"] <= c["settings"].tolerance


def test_boolean_impulse_request_is_not_a_physical_scalar():
    c = collect()
    with pytest.raises(ph.SolverFailure):
        require(c, expected=np.zeros((3, 3), bool))
