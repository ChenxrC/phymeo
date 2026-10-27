"""Experiment-only external boundary driven by a precomputed free trajectory.

Only node states are dynamic scene domains. The trajectory is immutable problem
data and its consumption index is explicit transactional history. This module
does not add a no-reaction mode to the public articulated physics program.
"""

from contextlib import ExitStack
from dataclasses import dataclass
from numbers import Integral
from time import perf_counter

import numpy as np

from ..constraints.node_robot import NodeRobotConstraints
from ..core.capabilities import SolverFailure
from ..core.robot import RobotInputs, RobotState, robot_array
from ..core.scene import (
    HistoryField,
    SceneInputs,
    SceneModel,
    SceneState,
    StageMetrics,
)
from ..core.state import ParticleState
from ..coupling.participant import RobotParticipantWorkspace
from ..runtime.program import ProgramCapabilities
from ..runtime.registry import StateRegistry
from ..runtime.scene_stepper import SceneStepper
from ..solvers.articulated_coupling import ArticulatedWorkspace
from ..solvers.robot import RobotWorkspace, make_robot_runtime
from ..validation.articulated import (
    EVENT_INTEGER,
    EVENT_SHAPES,
    PRESCRIBED_AUDIT_FIELDS,
    audit_slider_substep,
    compact_event,
    event_arrays,
    read_events,
)
from .articulated import (
    _arrays,
    _check_source_record,
    _frames,
    _record_state,
    _summary,
    articulated_scene,
)
from .rigid_cloth import SceneRun, SceneRunFailure, load_run, save_run


@dataclass(frozen=True, eq=False)
class PrescribedTrajectory:
    q: np.ndarray
    v: np.ndarray
    h: float

    def __post_init__(self):
        q = robot_array(self.q, "trajectory q", ndim=2, columns=1)
        v = robot_array(self.v, "trajectory v", ndim=2, columns=1)
        if q.shape != v.shape or len(q) < 2:
            raise ValueError("trajectory requires matching initial and later samples")
        if isinstance(self.h, (bool, np.bool_)) or not np.isfinite(self.h) or self.h <= 0:
            raise ValueError("trajectory time step must be positive and finite")
        q.flags.writeable = v.flags.writeable = False
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "h", float(self.h))


def free_robot_trajectory(fixture, *, substeps):
    if isinstance(substeps, (bool, np.bool_)) or not isinstance(substeps, Integral) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    model = fixture.program.participants["robot"].model
    q, v = np.empty((substeps + 1, 1)), np.empty((substeps + 1, 1))
    state = SceneState({"robot": fixture.initial.domains["robot"]})
    inputs = SceneInputs({"robot": fixture.inputs.domains["robot"]})
    q[0], v[0] = state.domains["robot"].q, state.domains["robot"].v
    with make_robot_runtime(model) as runtime:
        for index in range(substeps):
            result = runtime.step(state, inputs, fixture.config["h"])
            if not result.report.accepted:
                raise SolverFailure("free source trajectory failed: " + result.report.reason)
            state = result.state
            q[index + 1], v[index + 1] = state.domains["robot"].q, state.domains["robot"].v
    return PrescribedTrajectory(q, v, fixture.config["h"])


class _PrescribedPort:
    def __init__(self, geometry):
        self.geometry = geometry

    @property
    def domain(self):
        return self.geometry.domain

    @property
    def dtype(self):
        return self.geometry.dtype

    @property
    def indices(self):
        return self.geometry.indices

    def positions(self):
        return self.geometry.positions()

    def velocities(self):
        return self.geometry.velocities()

    def check_loads(self, value):
        loads = robot_array(value, "boundary loads", ndim=2, columns=3)
        if loads.shape != (len(self.indices), 3):
            raise ValueError("boundary load shape must match attachments")
        return loads

    def response(self, impulses):
        return np.zeros_like(self.check_loads(impulses))

    def apply_impulses(self, impulses):
        self.check_loads(impulses)
        return self.snapshot()

    def snapshot(self):
        return self.geometry.snapshot()

    def validate_candidate(self, state):
        candidate = self.geometry.validate_candidate(state)
        if not np.array_equal(candidate.v, self.snapshot().v):
            raise ValueError("external boundary velocity cannot respond to contact")
        return candidate


class _BoundaryWorkspace(RobotParticipantWorkspace):
    """Reuse geometry mappings only; never invoke dynamic robot prediction."""

    def __init__(self, model, q, v, h):
        sample = RobotState(q, v)
        native = RobotWorkspace(model, "driver", sample, RobotInputs.zeros(model.dof_count), h)
        native.candidate = RobotState(q, v)
        super().__init__(native)
        self.prediction_ready = False

    def predict(self):
        if self.prediction_ready:
            raise RuntimeError("external boundary prediction already prepared")
        self.prediction_ready = True

    def point_port(self, indices, local_points=None):
        return _PrescribedPort(super().point_port(indices, local_points))

    def apply_position_load(self, indices, local_points, loads):
        self.point_port(indices, local_points).check_loads(loads)

    def apply_velocity_impulses(self, indices, local_points, impulses):
        self.point_port(indices, local_points).check_loads(impulses)

    def prepare_velocity(self):
        if self.velocity_ready:
            raise RuntimeError("external boundary velocity already prepared")
        self.velocity_ready = True

    def velocity_metrics(self):
        return StageMetrics()

    def diagnostics(self):
        return {
            "kind": "prescribed_external_boundary",
            "q": self.configuration(),
            "v": self.velocity(),
        }


class PrescribedArticulatedProgram:
    history_fields = (HistoryField("driver_index", (), np.int64),)

    def __init__(self, fixture, trajectory):
        if not isinstance(trajectory, PrescribedTrajectory):
            raise TypeError("expected PrescribedTrajectory")
        self.fixture = fixture
        self.trajectory = PrescribedTrajectory(trajectory.q, trajectory.v, trajectory.h)
        h, q, v = trajectory.h, trajectory.q, trajectory.v
        if h != fixture.config["h"]:
            raise ValueError("source time step must match the frozen scene")
        initial = fixture.initial.domains["robot"]
        if not np.array_equal(q[0], initial.q) or not np.array_equal(v[0], initial.v):
            raise ValueError("source trajectory initial state differs from free robot")
        mass = fixture.program.participants["robot"].model.mass[0]
        acceleration = fixture.inputs.domains["robot"].effort / mass
        if not np.allclose(v[1:], v[:-1] + h * acceleration, atol=5e-14, rtol=0) or not np.allclose(
            q[1:], q[:-1] + h * v[1:], atol=5e-14, rtol=0
        ):
            raise ValueError("source trajectory is not the frozen free robot solution")
        self.scene = SceneModel(
            tuple(domain for domain in fixture.program.scene.domains if domain.name != "robot")
        )
        self.initial = SceneState(
            {name: fixture.initial.domains[name] for name in fixture.node_names},
            {"driver_index": np.array(0, dtype=np.int64)},
        )
        self.inputs = SceneInputs(
            {name: fixture.inputs.domains[name] for name in fixture.node_names}
        )
        self.registry = StateRegistry(defaults=False)
        families = set()
        for name in fixture.node_names:
            family = fixture.program.participants[name].family
            if family.kind not in families:
                self.registry.register(family)
                families.add(family.kind)
        self.capabilities = ProgramCapabilities(
            "numpy", {"float64"}, families, {"prescribed_external_boundary"}, history="explicit"
        )

    def problem_signature(self):
        return {
            "law": "prescribed-free-robot-boundary-v1",
            "trajectory": self.trajectory,
            "node_scene": self.scene.identity,
            "geometry_source": self.fixture.program.problem_signature(),
        }

    def create_workspace(self, scene, state, inputs, h, settings, context):
        self.scene.assert_unchanged()
        if scene.identity != self.scene.identity:
            raise ValueError("prescribed program must use its bound node scene")
        if h != self.trajectory.h:
            raise ValueError("runtime time step must match the prescribed trajectory")
        index = int(state.history["driver_index"])
        if index < 0 or index >= len(self.trajectory.q):
            raise ValueError("driver index outside source trajectory")
        if index == len(self.trajectory.q) - 1:
            raise SolverFailure("external driver trajectory exhausted")
        return _PrescribedWorkspace(self, state, inputs, index, h, settings, context)

    def close(self):
        pass


class _PrescribedWorkspace(ArticulatedWorkspace):
    def __init__(self, program, state, inputs, index, h, settings, context):
        fixture = program.fixture
        self.node_names = fixture.node_names
        self.index = index
        self.history = {"driver_index": np.array(index + 1, dtype=np.int64)}
        self.participants = {}
        with ExitStack() as stack:
            driver = _BoundaryWorkspace(
                fixture.program.participants["robot"].model,
                program.trajectory.q[index + 1],
                program.trajectory.v[index + 1],
                h,
            )
            stack.callback(driver.close)
            self.participants["driver"] = driver
            for name in fixture.node_names:
                workspace = fixture.program.participants[name].create_workspace(
                    name, state.domains[name], inputs.domains[name], h, settings, context
                )
                stack.callback(workspace.close)
                self.participants[name] = workspace
            self.contact = NodeRobotConstraints(
                self.participants,
                "driver",
                fixture.program.attachments,
                fixture.program.contacts,
                h,
                settings,
            )
            self.resources = stack.pop_all()

    def result(self):
        return SceneState(
            {name: self.participants[name].result() for name in self.node_names}, self.history
        )


def make_prescribed_runtime(program, *, settings=None):
    if not isinstance(program, PrescribedArticulatedProgram):
        raise TypeError("expected PrescribedArticulatedProgram")
    return SceneStepper(
        program.scene,
        program,
        program.fixture.settings if settings is None else settings,
        registry=program.registry,
    )


def _prescribed_config(fixture, frames):
    return dict(
        fixture.config,
        interaction="prescribed_driver",
        frames=frames,
        duration=frames * fixture.config["dt"],
        dynamic_domains=list(fixture.node_names),
        driver={
            "source": "free RobotProgram trajectory",
            "history": "driver_index",
            "reaction": "absorbed by external driver; no dynamic robot domain",
        },
    )


def _prescribed_arrays(fixture, frames, trajectory):
    arrays = _arrays(fixture, frames)
    for field in ("q", "v"):
        arrays[f"driver_{field}"] = arrays.pop(f"robot_{field}")
        arrays[f"substep_driver_{field}"] = arrays.pop(f"substep_robot_{field}")
        arrays.pop(f"substep_predicted_robot_{field}")
        arrays[f"source_{field}"] = getattr(trajectory, field).copy()
    arrays["substep_driver_trajectory_error"] = arrays.pop("substep_robot_balance_error")
    arrays["driver_index"] = np.zeros(frames + 1, dtype=np.int64)
    arrays["substep_driver_index"] = np.empty(frames * 2, dtype=np.int64)
    arrays["driver_contact_impulse"] = np.empty((frames * 2, 3))
    arrays["driver_supplied_impulse"] = np.empty((frames * 2, 3))
    return arrays


def _contact_impulse(events, h):
    total = np.zeros(3)
    for event in events:
        total += event["robot_load"] / h if event["phase"] == 0 else event["robot_load"]
    return total


def run_prescribed_scene(*, scene="S2", frames=200, progress=None):
    frames = _frames(frames)
    fixture = articulated_scene(scene)
    trajectory = free_robot_trajectory(fixture, substeps=frames * 2)
    program = PrescribedArticulatedProgram(fixture, trajectory)
    arrays, events, stage_seconds = _prescribed_arrays(fixture, frames, trajectory), [], {}
    state = before = program.initial
    previous_driver = RobotState(trajectory.q[0], trajectory.v[0])
    start = perf_counter()
    with make_prescribed_runtime(program) as runtime:
        for frame in range(frames):
            result = runtime.step(state, program.inputs, fixture.config["dt"])
            if not result.report.accepted:
                raise SceneRunFailure(
                    f"prescribed {scene} frame {frame + 1}: {result.report.reason}"
                )
            for offset, (after, report) in enumerate(
                zip(result.substep_states, result.report.substeps, strict=True)
            ):
                index = frame * 2 + offset
                if after.history["driver_index"] != index + 1:
                    raise SceneRunFailure("driver consumption index disagrees with substep")
                diagnostic = report.diagnostics["participants"]["driver"]
                current_driver = RobotState(diagnostic["q"], diagnostic["v"])
                current_events = [
                    compact_event(event, fixture.node_names.index(event["domain"]), index)
                    for event in report.diagnostics["contact_events"]
                ]
                values, contacts = audit_slider_substep(
                    fixture,
                    before,
                    after,
                    current_events,
                    trajectory.q[index + 1],
                    trajectory.v[index + 1],
                    h=trajectory.h,
                    prescribed=(previous_driver, current_driver),
                )
                _record_state(arrays, after, index, "substep_")
                for field in ("q", "v"):
                    arrays[f"substep_driver_{field}"][index] = getattr(current_driver, field)
                arrays["substep_driver_index"][index] = after.history["driver_index"]
                impulse = _contact_impulse(current_events, trajectory.h)
                (
                    arrays["driver_contact_impulse"][index],
                    arrays["driver_supplied_impulse"][index],
                ) = impulse, -impulse
                for name, value in values.items():
                    arrays["substep_" + name][index] = value
                for name, value in contacts.items():
                    arrays[f"substep_{name}_contact"][index] = value
                for field in ("position_iterations", "velocity_iterations"):
                    arrays["substep_" + field][index] = getattr(report, field)
                arrays["substep_reported_residual"][index] = report.max_residual
                for name, seconds in report.timings.items():
                    stage_seconds[name] = stage_seconds.get(name, 0.0) + seconds
                events.extend(current_events)
                before, previous_driver = after, current_driver
            state = result.state
            _record_state(arrays, state, frame + 1)
            for field in ("q", "v"):
                arrays[f"driver_{field}"][frame + 1] = getattr(current_driver, field)
            arrays["driver_index"][frame + 1] = state.history["driver_index"]
            if progress is not None:
                progress(frame + 1, result.report)
    arrays.update(event_arrays(events))
    summary = _summary(fixture, arrays, frames, audit_fields=PRESCRIBED_AUDIT_FIELDS)
    if frames == 200 and any(count < 5 for count in summary["contact_substeps"].values()):
        raise SceneRunFailure("complete prescribed scene lacks actual node interactions")
    summary.update(
        end_to_end_seconds=perf_counter() - start,
        stage_seconds=stage_seconds,
        max_absorbed_impulse=float(np.max(np.abs(arrays["driver_contact_impulse"]))),
    )
    return SceneRun(_prescribed_config(fixture, frames), summary, arrays)


def validate_prescribed_run(run, *, require_complete=False):
    frames = _frames(run.config["frames"])
    fixture, arrays = articulated_scene(run.config["variant"]), run.arrays
    if run.config != _prescribed_config(fixture, frames):
        raise ValueError("prescribed trace configuration differs from frozen scene")
    if require_complete and frames != 200:
        raise ValueError("complete acceptance requires 200 frames")
    trajectory = PrescribedTrajectory(arrays["source_q"], arrays["source_v"], fixture.config["h"])
    program = PrescribedArticulatedProgram(fixture, trajectory)
    if len(trajectory.q) != frames * 2 + 1:
        raise ValueError("source horizon differs from recorded frame count")
    expected = _prescribed_arrays(fixture, frames, trajectory)
    count = len(arrays["event_phase"])
    expected.update(
        {
            "event_" + name: np.empty(
                (count, *shape), dtype=np.int64 if name in EVENT_INTEGER else np.float64
            )
            for name, shape in EVENT_SHAPES.items()
        }
    )
    if set(arrays) != set(expected) or any(
        arrays[name].shape != value.shape
        or arrays[name].dtype != value.dtype
        or not np.isfinite(arrays[name]).all()
        for name, value in expected.items()
    ):
        raise ValueError("prescribed trace shape, precision or fields mismatch")
    for name in (*fixture.topology, "time", "substep_time"):
        if not np.array_equal(arrays[name], expected[name]):
            raise ValueError("prescribed trace topology/time mismatch")
    if not np.array_equal(
        arrays["driver_index"], np.arange(0, frames * 2 + 1, 2)
    ) or not np.array_equal(arrays["substep_driver_index"], np.arange(1, frames * 2 + 1)):
        raise ValueError("prescribed trace history index mismatch")
    for field in ("q", "v"):
        if not np.array_equal(
            arrays[f"driver_{field}"], getattr(trajectory, field)[::2]
        ) or not np.array_equal(arrays[f"substep_driver_{field}"], getattr(trajectory, field)[1:]):
            raise ValueError("driver motion differs from its immutable source")
    for name in fixture.node_names:
        for field, value in vars(program.initial.domains[name]).items():
            if not np.array_equal(arrays[f"{name}_{field}"][0], value) or not np.array_equal(
                arrays[f"{name}_{field}"][1:], arrays[f"substep_{name}_{field}"][1::2]
            ):
                raise ValueError("node frames differ from actual substeps")
    if (
        np.any(arrays["event_substep"] < 0)
        or np.any(arrays["event_substep"] >= frames * 2)
        or np.any(np.diff(arrays["event_substep"]) < 0)
    ):
        raise ValueError("invalid event substep order")
    before = program.initial
    for index in range(frames * 2):
        after = SceneState(
            {
                name: ParticleState(
                    arrays[f"substep_{name}_position"][index],
                    arrays[f"substep_{name}_velocity"][index],
                )
                for name in fixture.node_names
            },
            {"driver_index": arrays["substep_driver_index"][index]},
        )
        events = read_events(arrays, index)
        values, contacts = audit_slider_substep(
            fixture,
            before,
            after,
            events,
            trajectory.q[index + 1],
            trajectory.v[index + 1],
            h=trajectory.h,
            prescribed=(
                RobotState(trajectory.q[index], trajectory.v[index]),
                RobotState(trajectory.q[index + 1], trajectory.v[index + 1]),
            ),
        )
        for name, value in values.items():
            if arrays["substep_" + name][index] != value:
                raise ValueError("prescribed physical audit differs from recomputation")
        for name, value in contacts.items():
            if arrays[f"substep_{name}_contact"][index] != value:
                raise ValueError("prescribed contact counts differ from loads")
        impulse = _contact_impulse(events, trajectory.h)
        if not np.array_equal(
            impulse, arrays["driver_contact_impulse"][index]
        ) or not np.array_equal(-impulse, arrays["driver_supplied_impulse"][index]):
            raise ValueError("external reaction impulse ledger mismatch")
        before = after
    for field in ("position_iterations", "velocity_iterations"):
        if np.any(arrays["substep_" + field] < 1) or np.any(arrays["substep_" + field] > 200):
            raise ValueError("invalid prescribed iteration evidence")
    residual = arrays["substep_reported_residual"]
    if np.any(residual < 0) or np.any(residual > fixture.settings.tolerance):
        raise ValueError("invalid prescribed residual evidence")
    summary = _summary(fixture, arrays, frames, audit_fields=PRESCRIBED_AUDIT_FIELDS)
    summary["max_absorbed_impulse"] = float(np.max(np.abs(arrays["driver_contact_impulse"])))
    if any(run.metrics.get(key) != value for key, value in summary.items()):
        raise ValueError("prescribed summary disagrees with numerical evidence")
    if frames == 200 and any(count < 5 for count in summary["contact_substeps"].values()):
        raise ValueError("complete prescribed horizon lacks physical interaction")
    return summary


def save_prescribed_run(run, destination, *, source=None):
    validate_prescribed_run(run)
    save_run(run, destination, source=source)


def load_prescribed_run(directory, *, require_complete=False):
    _check_source_record(directory)
    run = load_run(directory)
    validate_prescribed_run(run, require_complete=require_complete)
    return run
