"""Experimental API-closure probe; no automatic contact solver or production law.

Run modes serially: numpy, interpret, cuda. Each run performs exactly two trials.
Held p/h is replayed once; prescribed eta is applied in the live native velocity
stage before workspace close. Existing material-ledger equations stay unchanged.
"""

import argparse
import hashlib
import json
import platform
import traceback
from collections.abc import Mapping
from fractions import Fraction as F
from importlib.metadata import distribution
from pathlib import Path

import numpy as np

import phymeo as ph
from phymeo.experiments.rigid_cloth import provenance
from phymeo.geometry.robot_sphere import RobotSphereAttachments
from phymeo.runtime.load_replay import SceneLoadReplay
from phymeo.runtime.program import own_diagnostics
from phymeo.solvers.articulated_coupling import ArticulatedCouplingProgram


def analytic(chain):
    """Independently derive the stated examples in exact rational arithmetic."""
    h, eta, mass = F(1, 4), F(1, 3), F(2)
    p = F(3, 7) if chain else F(1, 3)
    x0 = [F(1, 8), F(9, 8)] if chain else [F(1, 8)]
    v = [F(-1) + p] + ([F(-1)] if chain else [])
    x = [a + h * b for a, b in zip(x0, v, strict=True)]
    multiplier = F(0)
    if chain:
        alpha = F(1, 16) / h**2
        multiplier = -(x[1] - x[0] - 1) / (2 + alpha)
        x[0] -= multiplier
        x[1] += multiplier
        v[0] -= multiplier / h
        v[1] += multiplier / h
    q, rv = -h * p / mass, -p / mass
    post_v, post_rv = [v[0] + eta] + v[1:], rv - eta / mass
    potential = (x[1] - x[0] - 1) ** 2 / (2 * F(1, 16)) if chain else F(0)
    kinetic = mass * post_rv**2 / 2 + sum(a * a / 2 for a in post_v)
    momentum = mass * post_rv + sum(post_v)
    if chain:
        assert x == [F(-3, 56), F(51, 56)] and q == F(-3, 56)
        assert v == [F(-5, 7), F(-6, 7)] and rv == F(-3, 14)
        assert post_v == [F(-8, 21), F(-6, 7)] and post_rv == F(-8, 21)
        assert potential == F(1, 98) and kinetic + potential == F(25, 42)
        assert momentum == -2
    else:
        assert x == [F(-1, 24)] and q == F(-1, 24)
        assert v == [F(-2, 3)] and rv == F(-1, 6)
        assert post_v == [F(-1, 3)] and post_rv == F(-1, 3)
        assert kinetic == F(1, 6) and momentum == -1
    return dict(
        h=h,
        eta=eta,
        p=p,
        stage1_x=x,
        stage1_q=q,
        stage1_v=v,
        stage1_robot_v=rv,
        post_v=post_v,
        post_robot_v=post_rv,
        material_multiplier=multiplier,
        potential=potential,
        kinetic=kinetic,
        total_energy=kinetic + potential,
        total_momentum=momentum,
    )


def owned_state(state):
    return ph.SceneState(
        {
            name: (
                ph.RobotState(value.q.copy(), value.v.copy())
                if isinstance(value, ph.RobotState)
                else value.copy()
            )
            for name, value in state.domains.items()
        },
        {name: value.copy() for name, value in state.history.items()},
    )


def state_data(state):
    robot, nodes = state.domains["robot"], state.domains["nodes"]
    return dict(
        robot_q=robot.q, robot_v=robot.v, node_position=nodes.position, node_velocity=nodes.velocity
    )


class ExperimentalImpulseWrapper:
    """Instrumentation only; eta is prescribed, never solved by this wrapper."""

    def __init__(self, base, book, expected, settings, context):
        self.base, self.book, self.expected = base, book, expected
        self.settings, self.context = settings, context
        self.workspaces = []

    @property
    def capabilities(self):
        return self.base.capabilities

    @property
    def history_fields(self):
        return self.base.history_fields

    def problem_signature(self):
        return dict(
            experiment="prescribed-live-terminal-normal-impulse-v1",
            base=self.base.problem_signature(),
            eta=str(self.expected["eta"]),
        )

    def create_workspace(self, scene, state, inputs, h, settings, context):
        assert settings.substeps == 1
        native = self.base.create_workspace(scene, state, inputs, h, settings, context)
        ws = ExperimentalImpulseWorkspace(self, native, inputs)
        self.workspaces.append(ws)
        return ws

    def close(self):
        self.base.close()


class ExperimentalImpulseWorkspace:
    def __init__(self, owner, native, inputs):
        self.owner, self.native, self.inputs = owner, native, inputs
        self.sweeps, self.calls, self.close_calls = 0, 0, 0
        self.before = self.after = self.evidence = None
        self.events = []

    def __getattr__(self, name):
        return getattr(self.native, name)

    def position_sweep(self):
        self.native.position_sweep()
        self.sweeps += 1

    def velocity_sweep(self):
        assert not self.close_calls
        self.native.velocity_sweep()
        assert self.native.velocity_metrics().max_residual <= self.owner.settings.tolerance
        if self.calls:
            return
        self.events.append("base_velocity_converged")
        self.before = owned_state(self.native.result())
        diagnostics = own_diagnostics(self.native.diagnostics())
        nodes = self.before.domains["nodes"]
        expected = np.zeros(nodes.position.shape, dtype=object)
        expected[0, 1] = self.owner.expected["p"]
        certified = self.owner.book.require(
            nodes,
            self.inputs.domains["nodes"],
            expected,
            diagnostics["participants"]["nodes"],
            sweeps=self.sweeps,
        )
        self.events.append("stage1_material_certified")
        eta = float(self.owner.expected["eta"])
        self.native.participants["nodes"].apply_velocity_impulses(
            np.array([0]), None, np.array([[0.0, eta, 0.0]])
        )
        self.native.participants["robot"].apply_velocity_impulses(
            np.array([0]), np.zeros((1, 3)), np.array([[0.0, -eta, 0.0]])
        )
        self.calls += 1
        self.events.append("live_prescribed_eta_applied")
        self.after = owned_state(self.native.result())
        after_diag = own_diagnostics(self.native.diagnostics())
        for name in ("robot", "nodes"):
            field = "q" if name == "robot" else "position"
            before = getattr(self.before.domains[name], field)
            after = getattr(self.after.domains[name], field)
            assert before.dtype == after.dtype and before.shape == after.shape
            assert before.tobytes() == after.tobytes(), "terminal impulse moved configuration"
        rejection = None
        try:
            self.owner.book.require(
                self.after.domains["nodes"],
                self.inputs.domains["nodes"],
                expected,
                diagnostics["participants"]["nodes"],
                sweeps=self.sweeps,
            )
        except ph.SolverFailure as error:
            rejection = str(error)
        assert rejection and "velocity" in rejection, "old reconstruction ledger must reject v+"
        self.evidence = dict(
            before=state_data(self.before),
            after=state_data(self.after),
            material_path=diagnostics["participants"]["nodes"]["material_path"],
            material_certificate=certified,
            original_ledger_post_rejection=rejection,
            participant_diagnostics_before=diagnostics["participants"],
            participant_diagnostics_after=after_diag["participants"],
            bitwise_configuration_preserved=True,
            terminal_impulse_calls=self.calls,
        )

    def diagnostics(self):
        return dict(self.native.diagnostics(), experimental_terminal_impulse=self.evidence)

    def close(self):
        self.close_calls += 1
        self.events.append("close")
        self.native.close()


def case(mode, chain, exact):
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
    constraints = (
        ph.DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([1 / 16]))
        if chain
        else None
    )
    nodes = ph.ParticleState(
        np.array([[0.0, 1 / 8, 0.0], [0.0, 9 / 8, 0.0]])[:n], np.tile([0.0, -1.0, 0.0], (n, 1))
    )
    initial = ph.SceneState({"robot": ph.RobotState(np.zeros(1), np.zeros(1)), "nodes": nodes})
    original = ph.SceneInputs({"robot": ph.RobotInputs.zeros(1), "nodes": ph.StepInputs.zeros(n)})
    scene = ph.SceneModel(
        (
            ph.DomainBinding("robot", "robot", model),
            ph.DomainBinding("nodes", "particles", particles),
        )
    )
    if mode == "numpy":
        participants = {
            "robot": ph.RobotParticipant(model),
            "nodes": ph.ClothParticipant(particles, constraints, record_material_path=True),
        }
        program_type = ArticulatedCouplingProgram
    else:
        from phymeo.backends.tyx_articulated import TyxArticulatedCouplingProgram

        participants = {
            "robot": ph.TyxRobotParticipant(model, mode),
            "nodes": ph.TyxClothParticipant(
                particles, constraints, mode, record_material_path=True
            ),
        }
        program_type = TyxArticulatedCouplingProgram
    attachments = RobotSphereAttachments(np.empty(0, dtype=int), np.empty((0, 3)), np.empty(0))
    base = program_type(scene, participants, attachments, ())
    context = ph.BackendContext(base.capabilities.backend)
    settings = ph.StepSettings(substeps=1, tolerance=1e-10 if mode == "numpy" else 2e-5)
    book = ph.MaterialPathLedger(
        particles,
        constraints,
        nodes,
        original.domains["nodes"],
        float(exact["h"]),
        settings,
        context,
    )
    wrapper = ExperimentalImpulseWrapper(base, book, exact, settings, context)
    node_integral = np.zeros((n, 3))
    node_integral[0, 1] = float(exact["p"])
    loads = {"nodes": {"force": node_integral}, "robot": {"effort": np.array([-float(exact["p"])])}}
    with SceneLoadReplay(
        scene,
        wrapper,
        initial,
        original,
        float(exact["h"]),
        channels={"nodes": ("force",), "robot": ("effort",)},
        settings=settings,
        context=context,
        registry=base.registry,
    ) as replay:
        trial = replay.evaluate(loads)
        assert trial.result.report.accepted, trial.result.report.reason
    assert len(wrapper.workspaces) == 1
    ws = wrapper.workspaces[0]
    assert ws.close_calls == ws.calls == 1
    assert ws.events == [
        "base_velocity_converged",
        "stage1_material_certified",
        "live_prescribed_eta_applied",
        "close",
    ]
    assert ws.native.contact.rows == () and not ws.native.contact.events
    atol = 2e-13 if mode == "numpy" else 2e-6
    before, after = ws.before, trial.result.state
    np.testing.assert_allclose(
        before.domains["nodes"].position[:, 1],
        [float(x) for x in exact["stage1_x"]],
        rtol=0,
        atol=atol,
    )
    np.testing.assert_allclose(
        before.domains["robot"].q, [float(exact["stage1_q"])], rtol=0, atol=atol
    )
    np.testing.assert_allclose(
        before.domains["nodes"].velocity[:, 1],
        [float(x) for x in exact["stage1_v"]],
        rtol=0,
        atol=atol,
    )
    np.testing.assert_allclose(
        before.domains["robot"].v, [float(exact["stage1_robot_v"])], rtol=0, atol=atol
    )
    np.testing.assert_allclose(
        after.domains["nodes"].velocity[:, 1],
        [float(x) for x in exact["post_v"]],
        rtol=0,
        atol=atol,
    )
    np.testing.assert_allclose(
        after.domains["robot"].v, [float(exact["post_robot_v"])], rtol=0, atol=atol
    )
    node_jump = after.domains["nodes"].velocity - before.domains["nodes"].velocity
    requested_jump = np.zeros((n, 3))
    requested_jump[0, 1] = float(exact["eta"])
    np.testing.assert_allclose(node_jump, requested_jump, rtol=0, atol=atol)
    np.testing.assert_allclose(
        after.domains["robot"].v - before.domains["robot"].v,
        [-float(exact["eta"]) / 2],
        rtol=0,
        atol=atol,
    )
    nv, rv = after.domains["nodes"].velocity, after.domains["robot"].v[0]
    kinetic = float(np.sum(nv**2) / 2 + rv**2)
    positions = after.domains["nodes"].position
    potential = (
        float((np.linalg.norm(positions[1] - positions[0]) - 1) ** 2 / (2 / 16)) if chain else 0.0
    )
    momentum = float(2 * rv + np.sum(nv[:, 1]))
    np.testing.assert_allclose(
        [momentum, kinetic, potential, kinetic + potential],
        [float(exact[k]) for k in ("total_momentum", "kinetic", "potential", "total_energy")],
        rtol=0,
        atol=atol,
    )
    np.testing.assert_allclose(
        trial.integrated_loads["nodes"]["force"], node_integral, rtol=0, atol=atol
    )
    np.testing.assert_allclose(
        trial.integrated_loads["robot"]["effort"], [-float(exact["p"])], rtol=0, atol=atol
    )
    post_diags = ws.evidence["participant_diagnostics_after"]
    if mode != "numpy":
        for name, diag in post_diags.items():
            old = ws.evidence["participant_diagnostics_before"][name]["execution"]
            execution = diag["execution"]
            assert execution["mode"] == mode and execution["compute_dtype"] == "float32"
            assert execution["completed_launches"] > old["completed_launches"] > 0
    assert trial.result.report.backend == context.backend
    np.testing.assert_array_equal(initial.domains["nodes"].velocity[:, 1], -np.ones(n))
    np.testing.assert_array_equal(initial.domains["robot"].v, np.zeros(1))
    return dict(
        case="distance_chain" if chain else "free_node",
        status="passed",
        analytic=exact,
        actual_backend=trial.result.report.backend,
        absolute_tolerance=atol,
        observed_total_momentum=momentum,
        observed_kinetic=kinetic,
        observed_potential=potential,
        observed_total_energy=kinetic + potential,
        integrated_loads=trial.integrated_loads,
        effective_inputs={
            "robot_effort": trial.effective_inputs.domains["robot"].effort,
            "node_force": trial.effective_inputs.domains["nodes"].force,
        },
        native_position_sweeps=ws.sweeps,
        workspace_close_calls=ws.close_calls,
        events=ws.events,
        evidence=ws.evidence,
    )


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, F):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def runtime_versions():
    """Report the installed distribution identity as well as the requested pin."""
    tyx_distribution = distribution("tyx-compute")
    direct_url = tyx_distribution.read_text("direct_url.json")
    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        tyx_distribution="tyx-compute",
        tyx_version=tyx_distribution.version,
        tyx_direct_url=json.loads(direct_url) if direct_url is not None else None,
        requested_tyx_pin=(Path(__file__).resolve().parents[1] / "requirements" / "tyx.txt")
        .read_text(encoding="utf-8")
        .strip(),
    )


def development_history():
    """Keep earlier probe wiring failures separate from final rerun evidence."""
    return dict(
        prior_numpy_wiring_failures=[
            dict(
                attempted_cases=2,
                passed_cases=0,
                error_type="TypeError",
                error="unsupported configuration type: Fraction",
                phase="configuration signature binding; before native workspace creation",
            ),
            dict(
                attempted_cases=2,
                passed_cases=0,
                error_type="AttributeError",
                error="'RobotState' object has no attribute 'copy'",
                phase="owned pre-impulse snapshot; before terminal impulse",
            ),
        ],
        corrected_initial_probe=dict(
            script_sha256="58a0a3a6b2db05d877fd13c3a16dda3a6713b894c23ebe50d8daa20569292515",
            passed_cases_by_mode={"numpy": 2, "interpret": 2, "cuda": 2},
            cases_per_mode=2,
            scope="Initial API-closure results before formatting and provenance enrichment; "
            "not the final rerun cases counted below",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("numpy", "interpret", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = dict(
        experimental=True,
        production_law=False,
        mode=args.mode,
        status="running",
        evidence_revision="formatted-provenance-rerun-v1",
        source=provenance(),
        runtime=runtime_versions(),
        development_history=development_history(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="API closure only: one held-load native replay and prescribed live terminal eta",
        excluded=[
            "automatic eta/p solution",
            "contact generation",
            "history/control proof",
            "CCD",
            "rotation",
            "performance",
            "production acceptance",
        ],
        passed_cases=0,
        attempted_cases=0,
        cases=[],
    )
    try:
        exacts = [analytic(False), analytic(True)]
        record["independent_fraction_verification"] = "2/2 passed before backend trials"
        for chain, exact in zip((False, True), exacts, strict=True):
            record["attempted_cases"] += 1
            try:
                record["cases"].append(case(args.mode, chain, exact))
                record["passed_cases"] += 1
            except Exception as error:
                record["cases"].append(
                    dict(
                        case="distance_chain" if chain else "free_node",
                        status="failed",
                        error_type=type(error).__name__,
                        error=str(error),
                        traceback=traceback.format_exc(),
                    )
                )
                traceback.print_exc()
        record["status"] = "passed" if record["passed_cases"] == 2 else "failed"
    except Exception as error:
        record.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        traceback.print_exc()
    record["source_after"] = provenance()
    record["script_sha256_after"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (
        record["source_after"]["source_sha256"] != record["source"]["source_sha256"]
        or record["script_sha256_after"] != record["script_sha256"]
    ):
        record["status"] = "source_changed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plain(record), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"{args.mode}: {record['passed_cases']}/2 passed; {record['status']}; {args.output}")
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
