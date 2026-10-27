"""Frozen S2/S3/S4 runs with actual substep states and independently audited loads."""

import json
import re
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np

from ..builders.cloth import cloth_grid
from ..builders.tetrahedral import tetrahedral_block
from ..constraints.node_robot import NodeRobotContact
from ..core.robot import RobotInputs, RobotModel, RobotState
from ..core.scene import DomainBinding, SceneInputs, SceneModel, SceneState, StepSettings
from ..core.state import ParticleState, StepInputs
from ..coupling.participant import ClothParticipant, RobotParticipant, TetrahedralParticipant
from ..geometry.robot_sphere import RobotSphereAttachments
from ..solvers.articulated_coupling import ArticulatedCouplingProgram, make_articulated_runtime
from ..validation.articulated import (
    AUDIT_FIELDS,
    EVENT_INTEGER,
    EVENT_SHAPES,
    audit_slider_substep,
    compact_event,
    event_arrays,
    read_events,
)
from .rigid_cloth import SceneRun, SceneRunFailure, load_run, save_run, trajectory_digest


@dataclass(frozen=True)
class ArticulatedScene:
    program: ArticulatedCouplingProgram
    initial: SceneState
    inputs: SceneInputs
    settings: StepSettings
    config: dict
    topology: dict
    node_names: tuple[str, ...]


def articulated_scene(name="S2"):
    if name not in ("S2", "S3", "S4"):
        raise ValueError("scene must be S2, S3 or S4")
    robot = RobotModel(
        parents=np.array([-1]),
        joint_types=("prismatic",),
        joint_position=np.array([[0.0, 0.06, 0]]),
        joint_orientation=np.array([[0.0, 0, 0, 1]]),
        joint_axis=np.array([[0.0, 1, 0]]),
        mass=np.array([0.5]),
        com=np.zeros((1, 3)),
        inertia_diagonal=np.full((1, 3), 0.01),
        gravity=np.zeros(3),
    )
    bindings = [DomainBinding("robot", "robot", robot)]
    participants = {"robot": RobotParticipant(robot)}
    states = {"robot": RobotState(np.zeros(1), np.array([-0.2]))}
    robot_loads = RobotInputs.zeros(1)
    robot_loads.effort[:] = -1
    inputs = {"robot": robot_loads}
    topology, contacts, centers, names = {}, [], [], []
    if name in ("S2", "S4"):
        patch = cloth_grid(
            3,
            3,
            spacing=0.05,
            density=0.4,
            structural_compliance=1e-4,
            shear_compliance=1e-4,
            pinned=(0, 2, 6, 8),
            origin=(-0.2, 0, -0.05),
        )
        model = replace(patch.model, gravity=np.zeros(3))
        participants["cloth"] = ClothParticipant(model, patch.constraints)
        bindings.append(DomainBinding("cloth", "particles", model))
        states["cloth"], inputs["cloth"] = patch.initial_state, StepInputs.zeros(9)
        names.append("cloth")
        contacts.append(NodeRobotContact("cloth", np.arange(9), np.full(9, len(centers))))
        centers.append([-0.15, 0, 0])
        topology.update(
            cloth_triangles=patch.triangles,
            cloth_edges=patch.constraints.indices,
            cloth_fixed=model.inverse_mass == 0,
            cloth_mass=patch.vertex_mass,
        )
    if name in ("S3", "S4"):
        model, state = tetrahedral_block(origin=(0.1, -0.1, -0.05), size=(0.1, 0.1, 0.1))
        participants["soft"] = TetrahedralParticipant(model)
        bindings.append(DomainBinding("soft", "tetrahedra", model))
        states["soft"], inputs["soft"] = state, StepInputs.zeros(18)
        names.append("soft")
        contacts.append(NodeRobotContact("soft", np.arange(18), np.full(18, len(centers))))
        centers.append([0.15, 0, 0])
        topology.update(
            soft_tetrahedra=model.tetrahedra,
            soft_fixed=model.fixed,
            soft_rest_position=model.rest_position,
            soft_mass=model.mass,
        )
    attachments = RobotSphereAttachments(
        np.zeros(len(centers), dtype=np.int64),
        np.array(centers, dtype=np.float64),
        np.full(len(centers), 0.05),
    )
    topology.update(
        attachment_links=attachments.link_indices,
        attachment_centers=attachments.local_centers,
        attachment_radii=attachments.radii,
    )
    program = ArticulatedCouplingProgram(
        SceneModel(tuple(bindings)), participants, attachments, contacts
    )
    config = {
        "scene": "P5-articulated-v1",
        "variant": name,
        "interaction": "two_way",
        "dt": 1 / 2000,
        "substeps": 2,
        "h": 1 / 4000,
        "dtype": "float64",
        "backend": "numpy",
        "position_iterations": 200,
        "velocity_iterations": 200,
        "tolerance": 1e-8,
        "reference_length": 1.0,
        "reference_speed": 1.0,
        "node_domains": names,
        "robot": {
            "root": [0, 0.06, 0],
            "joint": "Y-prismatic",
            "mass": 0.5,
            "inertia_diagonal": [0.01, 0.01, 0.01],
            "q0": 0.0,
            "v0": -0.2,
            "effort": -1.0,
            "gravity": [0, 0, 0],
        },
        "cloth": None
        if "cloth" not in names
        else {
            "shape": [3, 3],
            "spacing": 0.05,
            "density": 0.4,
            "compliance": 1e-4,
            "origin": [-0.2, 0, -0.05],
            "fixed": [0, 2, 6, 8],
            "gravity": [0, 0, 0],
        },
        "soft": None
        if "soft" not in names
        else {
            "shape": [3, 2, 3],
            "origin": [0.1, -0.1, -0.05],
            "size": [0.1, 0.1, 0.1],
            "density": 200,
            "young_modulus": 500,
            "poisson_ratio": 0.3,
            "gravity": [0, 0, 0],
            "fixed": "bottom_y",
        },
        "contact": "explicit discrete node/link sphere; frictionless; zero restitution",
    }
    return ArticulatedScene(
        program,
        SceneState(states),
        SceneInputs(inputs),
        StepSettings(substeps=2, tolerance=1e-8, record_substeps=True),
        config,
        topology,
        tuple(names),
    )


def _frames(value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not 1 <= value <= 200
    ):
        raise ValueError("frames must be an integer from 1 to the frozen 200-frame horizon")
    return int(value)


def _arrays(fixture, frames):
    substeps = frames * 2
    arrays = {name: value.copy() for name, value in fixture.topology.items()}
    arrays["time"] = np.arange(frames + 1) * fixture.config["dt"]
    arrays["substep_time"] = np.arange(1, substeps + 1) * fixture.config["h"]
    for name, state in fixture.initial.domains.items():
        for field, value in vars(state).items():
            arrays[f"{name}_{field}"] = np.empty((frames + 1, *value.shape))
            arrays[f"substep_{name}_{field}"] = np.empty((substeps, *value.shape))
            arrays[f"{name}_{field}"][0] = value
    for field in ("q", "v"):
        arrays[f"substep_predicted_robot_{field}"] = np.empty((substeps, 1))
    for name in AUDIT_FIELDS:
        arrays["substep_" + name] = np.empty(substeps)
    for name in (
        "position_iterations",
        "velocity_iterations",
        *[domain + "_contact" for domain in fixture.node_names],
    ):
        arrays["substep_" + name] = np.empty(substeps, dtype=np.int64)
    arrays["substep_reported_residual"] = np.empty(substeps)
    return arrays


def _record_state(arrays, state, index, prefix=""):
    for name, native in state.domains.items():
        for field, value in vars(native).items():
            arrays[f"{prefix}{name}_{field}"][index] = value


def _state_at(fixture, arrays, index):
    states = {
        "robot": RobotState(arrays["substep_robot_q"][index], arrays["substep_robot_v"][index])
    }
    states.update(
        {
            name: ParticleState(
                arrays[f"substep_{name}_position"][index], arrays[f"substep_{name}_velocity"][index]
            )
            for name in fixture.node_names
        }
    )
    return SceneState(states)


def _summary(fixture, arrays, frames, *, audit_fields=AUDIT_FIELDS):
    metrics = {
        "accepted_frames": frames,
        "audited_substeps": frames * 2,
        "complete_horizon": frames == 200,
        "contact_substeps": {
            name: int(np.count_nonzero(arrays[f"substep_{name}_contact"]))
            for name in fixture.node_names
        },
        "trajectory_sha256": trajectory_digest(arrays),
    }
    for field in audit_fields:
        reduction = np.min if field in ("min_det_f", "min_singular_value") else np.max
        key = (
            field
            if field in ("min_det_f", "min_singular_value", "max_singular_value")
            else "max_" + field
        )
        metrics[key] = float(reduction(arrays["substep_" + field]))
    return metrics


def run_articulated_scene(*, scene="S2", frames=200, progress=None):
    frames = _frames(frames)
    fixture = articulated_scene(scene)
    arrays, events, stage_seconds = _arrays(fixture, frames), [], {}
    state, before = fixture.initial, fixture.initial
    start = perf_counter()
    with make_articulated_runtime(fixture.program, settings=fixture.settings) as runtime:
        for frame in range(frames):
            result = runtime.step(state, fixture.inputs, fixture.config["dt"])
            if not result.report.accepted:
                raise SceneRunFailure(f"{scene} frame {frame + 1}: {result.report.reason}")
            for offset, (after, report) in enumerate(
                zip(result.substep_states, result.report.substeps, strict=True)
            ):
                index = frame * 2 + offset
                current_events = [
                    compact_event(event, fixture.node_names.index(event["domain"]), index)
                    for event in report.diagnostics["contact_events"]
                ]
                robot = report.diagnostics["participants"]["robot"]
                values, contact = audit_slider_substep(
                    fixture,
                    before,
                    after,
                    current_events,
                    robot["predicted_q"],
                    robot["predicted_v"],
                    h=fixture.config["h"],
                )
                _record_state(arrays, after, index, "substep_")
                for field in ("q", "v"):
                    arrays[f"substep_predicted_robot_{field}"][index] = robot["predicted_" + field]
                for field, value in values.items():
                    arrays["substep_" + field][index] = value
                for name, value in contact.items():
                    arrays[f"substep_{name}_contact"][index] = value
                for field in ("position_iterations", "velocity_iterations"):
                    arrays["substep_" + field][index] = getattr(report, field)
                arrays["substep_reported_residual"][index] = report.max_residual
                for phase, seconds in report.timings.items():
                    stage_seconds[phase] = stage_seconds.get(phase, 0.0) + seconds
                events.extend(current_events)
                before = after
            state = result.state
            _record_state(arrays, state, frame + 1)
            if progress is not None:
                progress(frame + 1, result.report)
    elapsed = perf_counter() - start
    arrays.update(event_arrays(events))
    metrics = _summary(fixture, arrays, frames)
    if frames == 200 and any(count < 5 for count in metrics["contact_substeps"].values()):
        raise SceneRunFailure(
            "complete scene must contain at least five contact substeps per node domain"
        )
    metrics.update(end_to_end_seconds=elapsed, stage_seconds=stage_seconds)
    return SceneRun(
        dict(fixture.config, frames=frames, duration=frames * fixture.config["dt"]), metrics, arrays
    )


def validate_articulated_run(run, *, require_complete=False):
    frames = _frames(run.config["frames"])
    fixture = articulated_scene(run.config["variant"])
    config = dict(fixture.config, frames=frames, duration=frames * fixture.config["dt"])
    if run.config != config:
        raise ValueError("trace configuration differs from the frozen physical scene")
    if require_complete and frames != 200:
        raise ValueError("complete acceptance requires 200 frames")
    expected = _arrays(fixture, frames)
    arrays = run.arrays
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
        raise ValueError("trace shape, precision or field schema mismatch")
    for name, value in fixture.topology.items():
        if not np.array_equal(arrays[name], value):
            raise ValueError("trace topology differs from the frozen scene")
    if not np.array_equal(arrays["time"], expected["time"]) or not np.array_equal(
        arrays["substep_time"], expected["substep_time"]
    ):
        raise ValueError("trace time mismatch")
    for name, state in fixture.initial.domains.items():
        for field, value in vars(state).items():
            if not np.array_equal(arrays[f"{name}_{field}"][0], value) or not np.array_equal(
                arrays[f"{name}_{field}"][1:], arrays[f"substep_{name}_{field}"][1::2]
            ):
                raise ValueError("frame trace does not match actual substeps")
    if (
        np.any(arrays["event_substep"] < 0)
        or np.any(arrays["event_substep"] >= frames * 2)
        or np.any(np.diff(arrays["event_substep"]) < 0)
    ):
        raise ValueError("event substep ordering mismatch")
    before = fixture.initial
    for index in range(frames * 2):
        after = _state_at(fixture, arrays, index)
        values, contacts = audit_slider_substep(
            fixture,
            before,
            after,
            read_events(arrays, index),
            arrays["substep_predicted_robot_q"][index],
            arrays["substep_predicted_robot_v"][index],
            h=fixture.config["h"],
        )
        for name, value in values.items():
            if arrays["substep_" + name][index] != value:
                raise ValueError("saved physical audit differs from independent recomputation")
        for name, value in contacts.items():
            if arrays[f"substep_{name}_contact"][index] != value:
                raise ValueError("saved contact count differs from recorded loads")
        before = after
    for field in ("position_iterations", "velocity_iterations"):
        if np.any(arrays["substep_" + field] < 1) or np.any(arrays["substep_" + field] > 200):
            raise ValueError("invalid iteration evidence")
    residual = arrays["substep_reported_residual"]
    if np.any(residual < 0) or np.any(residual > fixture.settings.tolerance):
        raise ValueError("invalid residual evidence")
    summary = _summary(fixture, arrays, frames)
    if any(run.metrics.get(key) != value for key, value in summary.items()):
        raise ValueError("trace summary differs from numerical evidence")
    if frames == 200 and any(count < 5 for count in summary["contact_substeps"].values()):
        raise ValueError("complete horizon lacks physical interaction")
    return summary


def save_articulated_run(run, destination, *, source=None):
    validate_articulated_run(run)
    save_run(run, destination, source=source)


def _check_source_record(directory):
    record = json.loads((Path(directory) / "run.json").read_text(encoding="utf-8"))
    source = record.get("provenance", {})
    if not re.fullmatch(r"[0-9a-f]{64}", source.get("source_sha256", "")) or not all(
        source.get(key) for key in ("python", "numpy", "platform")
    ):
        raise ValueError("missing or malformed source provenance")


def load_articulated_run(directory, *, require_complete=False):
    _check_source_record(directory)
    run = load_run(directory)
    validate_articulated_run(run, require_complete=require_complete)
    return run
