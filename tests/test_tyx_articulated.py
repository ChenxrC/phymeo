"""Actual backend coupling, independent impulse balance and global rollback."""

import importlib
import importlib.util
from dataclasses import replace

import numpy as np
import pytest
from test_articulated_coupling import case as cpu_case
from test_articulated_coupling import pairs, spheres

from phymeo.backends.tyx_participants import (
    TyxClothParticipant,
    TyxRobotParticipant,
    TyxTetrahedralParticipant,
)
from phymeo.builders.tetrahedral import tetrahedral_block
from phymeo.constraints.distance import DistanceConstraints
from phymeo.core.capabilities import SolverFailure
from phymeo.core.scene import (
    DomainBinding,
    HistoryField,
    SceneInputs,
    SceneModel,
    SceneState,
    StepSettings,
)
from phymeo.core.state import StepInputs

pytestmark = pytest.mark.tyx


def api():
    name = "phymeo.backends.tyx_articulated"
    assert importlib.util.find_spec(name) is not None, "P6 device coupling missing"
    return importlib.import_module(name)


def case(mode, **kwargs):
    reference, state, inputs = cpu_case(**kwargs)
    factories = {
        name: TyxRobotParticipant(p.model, mode)
        if name == "robot"
        else TyxClothParticipant(p.model, p.program.constraints, mode)
        for name, p in reference.participants.items()
    }
    program = api().TyxArticulatedCouplingProgram(
        reference.scene, factories, reference.attachments, reference.contacts
    )
    return program, state, inputs


def run(program, state, inputs, dt=0.01, **settings):
    with api().make_tyx_articulated_runtime(
        program, settings=StepSettings(substeps=1, tolerance=2e-6, **settings)
    ) as runtime:
        return runtime.step(state, inputs, dt)


def assert_restored(before, after):
    for name, state in before.domains.items():
        for field, original in vars(state).items():
            np.testing.assert_array_equal(getattr(after.domains[name], field), original)
            assert not np.shares_memory(getattr(after.domains[name], field), original)
    for name, value in before.history.items():
        np.testing.assert_array_equal(after.history[name], value)


def test_analytic_configuration_load_and_reaction_once(tyx_mode):
    program, state, inputs = case(tyx_mode)
    result = run(program, state, inputs)
    assert result.report.accepted, result.report.reason
    np.testing.assert_allclose(result.state.domains["robot"].q, [-0.01 / 3], atol=2e-5)
    np.testing.assert_allclose(result.state.domains["robot"].v, [-1 / 3], atol=2e-5)
    np.testing.assert_allclose(
        result.state.domains["nodes"].velocity, [[0.0, 2 / 3, 0.0]], atol=2e-5
    )
    diag = result.report.substeps[0].diagnostics
    assert diag["compute_dtype"] == "float32" and diag["backend"] == "tyx:" + tyx_mode
    assert diag["execution"]["completed_launches"] > 0
    assert diag["contact_events"]
    for event in diag["contact_events"]:
        np.testing.assert_array_equal(event["node_load"], -event["robot_load"])
        np.testing.assert_allclose(
            event["robot_generalized_load"], [event["robot_load"][1]], atol=2e-7
        )
        np.testing.assert_array_equal(event["robot_velocity_delta"], [0.0])
    np.testing.assert_array_equal(state.domains["robot"].q, [0.0])


def test_analytic_inelastic_velocity_and_shared_robot_mass(tyx_mode):
    program, state, inputs = case(tyx_mode, node_y=0.102, node_v=-0.2)
    result = run(program, state, inputs)
    assert result.report.accepted, result.report.reason
    np.testing.assert_allclose(result.state.domains["robot"].v, [-0.2 / 3], atol=2e-5)
    np.testing.assert_allclose(
        result.state.domains["nodes"].velocity, [[0.0, -0.2 / 3, 0.0]], atol=2e-5
    )
    program, state, inputs = case(tyx_mode, domains=2)
    result = run(program, state, inputs)
    assert result.report.accepted, result.report.reason
    np.testing.assert_allclose(result.state.domains["robot"].q, [-0.005], atol=2e-5)
    for name in ("nodes", "nodes1"):
        np.testing.assert_allclose(
            result.state.domains[name].position, [[0.0, 0.095, 0.0]], atol=2e-5
        )


@pytest.mark.parametrize("phase", ["position", "velocity"])
@pytest.mark.parametrize("fail_index", [2, 3])
def test_later_actual_iteration_budget_failure_restores_global_history(phase, fail_index, tyx_mode):
    program, state, inputs = case(
        tyx_mode,
        domains=2,
        node_y=0.102 if phase == "velocity" else 0.09,
        node_v=-0.2 if phase == "velocity" else 0.0,
    )
    program.history_fields = (HistoryField("index", (), np.int64),)
    program.capabilities = replace(program.capabilities, history="explicit")
    state = SceneState(state.domains, {"index": np.array(0, dtype=np.int64)})
    original, count = program.create_workspace, [0]

    def create(scene, before, *args):
        count[0] += 1
        ws = original(scene, before, *args)
        saved_result = ws.result

        def result():
            after = saved_result()
            return SceneState(after.domains, {"index": before.history["index"] + 1})

        ws.result = result
        if count[0] == fail_index:
            # Actual coupled contact requires >1 sweep; replace this later
            # candidate with the initial penetrating/closing input and use the
            # runtime's real iteration budget, rather than faking metrics.
            for name, participant in ws.participants.items():
                participant.initial = state.domains[name]
            ws.contact.settings = replace(ws.contact.settings, tolerance=1e-12)
        return ws

    program.create_workspace = create
    settings = StepSettings(
        substeps=3,
        position_iterations=1 if phase == "position" else 200,
        velocity_iterations=1 if phase == "velocity" else 200,
        tolerance=2e-6,
        record_substeps=True,
    )
    # Earlier substeps must actually be acceptable: defer contact rows until
    # the selected substep, then restore the real rows in the selected phase.
    original_create = program.create_workspace

    def scheduled(*args):
        ws = original_create(*args)
        if count[0] < fail_index:
            ws.contact.rows = ()
            ws.contact.position_load = np.zeros(0)
            ws.contact.velocity_impulse = np.zeros(0)
        return ws

    program.create_workspace = scheduled
    with api().make_tyx_articulated_runtime(program, settings=settings) as runtime:
        result = runtime.step(state, inputs, 0.03)
        assert not result.report.accepted
        assert result.report.completed_substeps == fail_index - 1
        assert_restored(state, result.state)
        assert result.substep_states == ()


@pytest.mark.parametrize("failure", ["kernel", "sync", "download", "close"])
@pytest.mark.parametrize("fail_index", [2, 3])
def test_late_device_boundary_failure_restores_and_runtime_can_retry(failure, fail_index, tyx_mode):
    program, state, inputs = case(tyx_mode, domains=2)
    program.history_fields = (HistoryField("index", (), np.int64),)
    program.capabilities = replace(program.capabilities, history="explicit")
    state = SceneState(state.domains, {"index": np.array(0, dtype=np.int64)})
    original, count, closed = program.create_workspace, [0], []

    def create(scene, before, *args):
        count[0] += 1
        index = count[0]
        ws = original(scene, before, *args)
        result, close = ws.result, ws.close

        def result_with_history():
            after = result()
            return SceneState(after.domains, {"index": before.history["index"] + 1})

        def close_and_record():
            close()
            closed.append(index)
            if index == fail_index and failure == "close":
                raise SolverFailure("injected close failure")

        ws.result, ws.close = result_with_history, close_and_record
        if index == fail_index and failure != "close":
            engine = ws.participants["nodes1"].engine
            call = engine.call

            def fail_after_real_kernel(*a, **kw):
                call(*a, **kw)
                # Numerical kernels and T1 resource tests are separate evidence;
                # this probe checks late-boundary transaction ownership.
                raise SolverFailure("injected " + failure + " boundary")

            engine.call = fail_after_real_kernel
        return ws

    program.create_workspace = create
    with api().make_tyx_articulated_runtime(
        program, settings=StepSettings(substeps=3, tolerance=2e-6, record_substeps=True)
    ) as runtime:
        result = runtime.step(state, inputs, 0.003)
        assert not result.report.accepted and result.report.completed_substeps == fail_index - 1
        assert closed == list(range(1, fail_index + 1))
        assert result.substep_states == ()
        assert_restored(state, result.state)
        retry = runtime.step(state, inputs, 0.003)
        assert retry.report.accepted, retry.report.reason
        np.testing.assert_array_equal(retry.state.history["index"], 3)


def test_partial_workspace_construction_closes_prior_runtime(tyx_mode):
    program, state, inputs = case(tyx_mode, domains=2)
    first, last = program.participants["robot"], program.participants["nodes1"]
    create, captured = first.create_workspace, []

    def first_create(*args):
        ws = create(*args)
        captured.append(ws)
        return ws

    def fail(*args):
        raise SolverFailure("partial construction")

    first.create_workspace, last.create_workspace = first_create, fail
    result = run(program, state, inputs)
    assert not result.report.accepted
    assert_restored(state, result.state)
    assert len(captured) == 1 and captured[0].engine._runtime.closed


def test_actual_short_cloth_and_tetrahedral_contact(tyx_mode):
    program, state, inputs = case(tyx_mode)
    model, soft = tetrahedral_block([-0.05, 0.0, -0.05], [0.1, 0.1, 0.1])
    soft.position[~model.fixed, 1] *= 0.98
    robot = replace(
        program.participants["robot"].model, joint_position=np.array([[0.0, 0.15, 0.0]])
    )
    combined = api().TyxArticulatedCouplingProgram(
        SceneModel(
            (DomainBinding("robot", "robot", robot), DomainBinding("soft", "tetrahedra", model))
        ),
        {
            "robot": TyxRobotParticipant(robot, tyx_mode),
            "soft": TyxTetrahedralParticipant(model, tyx_mode),
        },
        spheres(radii=(0.052,)),
        (pairs("soft", tuple(range(18)), (0,) * 18),),
    )
    result = run(
        combined,
        SceneState({"robot": state.domains["robot"], "soft": soft}),
        SceneInputs({"robot": inputs.domains["robot"], "soft": StepInputs.zeros(18)}),
        dt=0.00025,
    )
    assert result.report.accepted, result.report.reason
    diagnostic = result.report.substeps[0].diagnostics
    assert diagnostic["contact_events"]
    assert np.linalg.norm(diagnostic["participants"]["soft"]["internal_force"]) > 0
    np.testing.assert_array_equal(
        result.state.domains["soft"].position[model.fixed], soft.position[model.fixed]
    )
    # Two coincident cloth nodes joined by a compliant edge exercise actual
    # cloth constitutive sweeps as well as their shared contact reaction.
    from phymeo.core.model import ParticleModel
    from phymeo.core.state import ParticleState

    cloth = ParticleModel(np.ones(2), np.zeros(3))
    edges = DistanceConstraints(np.array([[0, 1]]), np.array([0.02]), np.array([1e-4]))
    robot = program.participants["robot"].model
    cloth_program = api().TyxArticulatedCouplingProgram(
        SceneModel(
            (DomainBinding("robot", "robot", robot), DomainBinding("cloth", "particles", cloth))
        ),
        {
            "robot": TyxRobotParticipant(robot, tyx_mode),
            "cloth": TyxClothParticipant(cloth, edges, tyx_mode),
        },
        spheres(),
        (pairs("cloth", (0, 1), (0, 0)),),
    )
    cloth_state = SceneState(
        {
            "robot": state.domains["robot"],
            "cloth": ParticleState(
                np.array([[-0.01, 0.095, 0.0], [0.01, 0.095, 0.0]]), np.zeros((2, 3))
            ),
        }
    )
    result = run(
        cloth_program,
        cloth_state,
        SceneInputs({"robot": inputs.domains["robot"], "cloth": StepInputs.zeros(2)}),
        dt=0.001,
    )
    assert result.report.accepted, result.report.reason
    assert result.report.substeps[0].diagnostics["contact_events"]


def test_contact_norm_preserves_tiny_normal_components_without_squaring_underflow(
    tyx_mode, tmp_path
):
    from phymeo.backends._tyx_contact_kernels import node_sphere_geometry
    from phymeo.backends.tyx_math import TyxMath

    with TyxMath(tyx_mode, cache_root=tmp_path) as engine:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            normal, gap, failed = engine.call(
                node_sphere_geometry,
                0.1,
                np.array([1e-25, 0.1, 0.0], np.float32),
                np.zeros(3, np.float32),
                np.zeros(3, np.float32),
                np.zeros(1, np.float32),
                np.zeros(1, np.int32),
                outputs=(3, 4, 5),
            )
        assert failed[0] == 0 and gap[0] == 0
        np.testing.assert_allclose(normal, [1e-24, 1.0, 0.0], atol=0.0, rtol=2e-5)


def test_contact_dot_bounds_omitted_subnormal_terms_and_rejects_lost_entire_result(
    tyx_mode, tmp_path
):
    from phymeo.backends.tyx_math import TyxMath

    contact = api().TyxNodeRobotConstraints.__new__(api().TyxNodeRobotConstraints)
    with TyxMath(tyx_mode, cache_root=tmp_path) as engine:
        contact.engine = engine
        with np.errstate(over="raise", invalid="raise", under="raise", divide="raise"):
            assert contact._dot(
                np.array([1e-23, -1.0, 0.0]), np.array([1e-21, -40.5, 0.0])
            ) == pytest.approx(40.5)
            with pytest.raises(SolverFailure, match="dot"):
                contact._dot(np.array([1e-23, 0.0, 0.0]), np.array([1e-21, 0.0, 0.0]))


@pytest.mark.parametrize("fail_index", [2, 3])
def test_late_actual_material_domain_failure_restores_and_retries(fail_index, tyx_mode):
    from test_tetrahedral_material import unit_tetra

    from phymeo.core.state import ParticleState

    base, state, inputs = case(tyx_mode)
    model = unit_tetra()
    robot = base.participants["robot"].model
    program = api().TyxArticulatedCouplingProgram(
        SceneModel(
            (DomainBinding("robot", "robot", robot), DomainBinding("soft", "tetrahedra", model))
        ),
        {
            "robot": TyxRobotParticipant(robot, tyx_mode),
            "soft": TyxTetrahedralParticipant(model, tyx_mode),
        },
        spheres(),
        (),
    )
    program.history_fields = (HistoryField("index", (), np.int64),)
    program.capabilities = replace(program.capabilities, history="explicit")
    state = SceneState(
        {
            "robot": state.domains["robot"],
            "soft": ParticleState(model.rest_position, np.zeros((4, 3))),
        },
        {"index": np.array(0, dtype=np.int64)},
    )
    inputs = SceneInputs({"robot": inputs.domains["robot"], "soft": StepInputs.zeros(4)})
    original, count = program.create_workspace, [0]

    def create(scene, before, *args):
        count[0] += 1
        ws = original(scene, before, *args)
        saved = ws.result

        def result():
            after = saved()
            return SceneState(after.domains, {"index": before.history["index"] + 1})

        ws.result = result
        if count[0] == fail_index:
            # Real explicit prediction drives a tetrahedron outside the frozen
            # material domain; validation must reject this later substep.
            ws.participants["soft"].inputs.force[1, 0] = -1e6
        return ws

    program.create_workspace = create
    with api().make_tyx_articulated_runtime(
        program, settings=StepSettings(substeps=3, tolerance=2e-6, record_substeps=True)
    ) as runtime:
        result = runtime.step(state, inputs, 0.003)
        assert not result.report.accepted and "domain" in result.report.reason
        assert result.report.completed_substeps == fail_index - 1
        assert_restored(state, result.state)
        assert result.substep_states == ()
        retry = runtime.step(state, inputs, 0.003)
        assert retry.report.accepted, retry.report.reason
        assert retry.state.history["index"] == 3


@pytest.mark.parametrize("bad", ["missing", "mode", "model", "node", "sphere", "cpu"])
def test_device_program_rejects_invalid_binding_without_fallback(bad, tyx_mode):
    program, _, _ = case(tyx_mode)
    factories = dict(program.participants)
    contacts = program.contacts
    if bad == "missing":
        factories.pop("nodes")
    elif bad == "mode":
        mode = "cuda" if tyx_mode != "cuda" else "interpret"
        factories["nodes"] = TyxClothParticipant(factories["nodes"].model, None, mode)
    elif bad == "model":
        from phymeo.core.model import ParticleModel

        factories["nodes"] = TyxClothParticipant(
            ParticleModel(np.array([2.0]), np.zeros(3)), None, tyx_mode
        )
    elif bad == "node":
        contacts = (pairs(nodes=(4,)),)
    elif bad == "sphere":
        contacts = (pairs(attachments=(4,)),)
    else:
        from phymeo.coupling.participant import ClothParticipant

        factories["nodes"] = ClothParticipant(factories["nodes"].model, None)
    with pytest.raises((ValueError, TypeError)):
        api().TyxArticulatedCouplingProgram(program.scene, factories, program.attachments, contacts)
