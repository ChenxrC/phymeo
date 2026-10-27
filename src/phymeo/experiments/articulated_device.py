"""Explicit P6 device/reference traces with independent auditing and execution evidence."""

from dataclasses import replace
from time import perf_counter

import numpy as np

from ..backends.tyx_articulated import TyxArticulatedCouplingProgram, make_tyx_articulated_runtime
from ..backends.tyx_participants import (
    TyxClothParticipant,
    TyxRobotParticipant,
    TyxTetrahedralParticipant,
)
from ..solvers.articulated_coupling import make_articulated_runtime
from ..validation.articulated import (
    EVENT_INTEGER,
    EVENT_SHAPES,
    compact_event,
    event_arrays,
    read_events,
)
from ..validation.articulated_device import audit_device_slider_substep
from .articulated import _arrays as _base_arrays
from .articulated import (
    _check_source_record,
    _frames,
    _record_state,
    _state_at,
    _summary,
    articulated_scene,
)
from .rigid_cloth import SceneRun, SceneRunFailure, load_run, save_run

MODES = ("interpret", "cpu_cpp", "cuda")
EXECUTION_FIELDS = ("launches", "completed_launches", "uploads", "downloads", "mode_code")
MODE_CODES = {"reference": 0, "interpret": 1, "cpu_cpp": 2, "cuda": 3}
TYX_SHA = "99f6df7c0fd8fdeef77f97df83914da27b64b7ee"


def _fixture(scene, mode):
    if mode not in MODE_CODES:
        raise ValueError("explicit P6 mode must be interpret/cpu_cpp/cuda/reference")
    fixture = articulated_scene(scene)
    program = fixture.program
    if mode != "reference":
        participants = {}
        for name, participant in program.participants.items():
            if name == "robot":
                participants[name] = TyxRobotParticipant(participant.model, mode)
            elif participant.family.kind == "tetrahedra":
                participants[name] = TyxTetrahedralParticipant(participant.model, mode)
            else:
                participants[name] = TyxClothParticipant(
                    participant.model, participant.program.constraints, mode
                )
        program = TyxArticulatedCouplingProgram(
            program.scene, participants, program.attachments, program.contacts
        )
    config = dict(
        fixture.config,
        scene="P6-articulated-v1",
        mode=mode,
        backend="numpy" if mode == "reference" else "tyx:" + mode,
        storage_dtype="float64",
        compute_dtype="float64" if mode == "reference" else "float32",
        host_validation="float64",
        ordered_serial=True,
        tolerance=2e-6,
        tyx_sha=None if mode == "reference" else TYX_SHA,
    )
    return replace(
        fixture, program=program, settings=replace(fixture.settings, tolerance=2e-6), config=config
    )


def articulated_device_scene(scene="S2", mode="cuda"):
    if mode not in MODES:
        raise ValueError("device mode must be interpret/cpu_cpp/cuda; reference is explicit")
    return _fixture(scene, mode)


def _arrays(fixture, frames):
    arrays = _base_arrays(fixture, frames)
    arrays.update(
        {"substep_" + field: np.zeros(frames * 2, dtype=np.int64) for field in EXECUTION_FIELDS}
    )
    return arrays


def _record_execution(arrays, index, diagnostics, mode):
    arrays["substep_mode_code"][index] = MODE_CODES[mode]
    if mode == "reference":
        return
    if diagnostics["backend"] != "tyx:" + mode or diagnostics["compute_dtype"] != "float32":
        raise SceneRunFailure("device returned incompatible execution metadata")
    for field in EXECUTION_FIELDS[:-1]:
        value = diagnostics["execution"][field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SceneRunFailure("malformed actual execution count")
        arrays["substep_" + field][index] = value


def _validate_execution(arrays, mode):
    if np.any(arrays["substep_mode_code"] != MODE_CODES[mode]):
        raise ValueError("recorded execution mode differs from configured mode")
    values = [arrays["substep_" + name] for name in EXECUTION_FIELDS[:-1]]
    launches, completed, uploads, downloads = values
    if any(np.any(value < 0) for value in values):
        raise ValueError("negative execution count")
    if mode == "reference":
        if any(np.any(value != 0) for value in values):
            raise ValueError("reference cannot claim device launches")
    elif np.any(completed <= 0) or not np.array_equal(launches, completed):
        raise ValueError("missing successful actual kernel execution/launch evidence")
    elif mode == "cuda":
        if np.any(uploads <= 0) or np.any(downloads <= 0):
            raise ValueError("CUDA missing device transfer evidence")
    elif np.any(uploads != 0) or np.any(downloads != 0):
        raise ValueError("CPU mode cannot claim CUDA transfers")


def _runtime(fixture, mode):
    if mode == "reference":
        return make_articulated_runtime(fixture.program, settings=fixture.settings)
    import importlib.metadata
    import json

    distribution = importlib.metadata.distribution("tyx-compute")
    source = json.loads(distribution.read_text("direct_url.json") or "{}")
    if source.get("vcs_info", {}).get("commit_id") != TYX_SHA:
        raise ValueError("P6 device experiment requires the frozen tyx source revision")
    return make_tyx_articulated_runtime(fixture.program, settings=fixture.settings)


def run_articulated_device(*, scene="S2", mode="cuda", frames=200, progress=None):
    if mode not in MODES:
        raise ValueError("device mode must be interpret/cpu_cpp/cuda")
    return _run(scene=scene, mode=mode, frames=frames, progress=progress)


def run_articulated_reference(*, scene="S2", frames=200, progress=None):
    return _run(scene=scene, mode="reference", frames=frames, progress=progress)


def _run(*, scene, mode, frames, progress):
    frames = _frames(frames)
    fixture = _fixture(scene, mode)
    arrays, events, stage_seconds = _arrays(fixture, frames), [], {}
    state, before = fixture.initial, fixture.initial
    start = perf_counter()
    with _runtime(fixture, mode) as runtime:
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
                _record_execution(arrays, index, report.diagnostics, mode)
                robot = report.diagnostics["participants"]["robot"]
                values, contact = audit_device_slider_substep(
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
    _validate_execution(arrays, mode)
    metrics = _summary(fixture, arrays, frames)
    if frames == 200 and any(count < 5 for count in metrics["contact_substeps"].values()):
        raise SceneRunFailure(
            "complete scene must contain at least five contact substeps per node domain"
        )
    metrics.update(end_to_end_seconds=elapsed, stage_seconds=stage_seconds)
    return SceneRun(
        dict(fixture.config, frames=frames, duration=frames * fixture.config["dt"]), metrics, arrays
    )


def validate_articulated_device_run(
    run, *, require_complete=False, expected_scene=None, expected_mode=None
):
    if expected_scene is not None and run.config.get("variant") != expected_scene:
        raise ValueError("trace scene does not match expected scene")
    if expected_mode is not None and run.config.get("mode") != expected_mode:
        raise ValueError("trace mode does not match expected execution")
    frames = _frames(run.config["frames"])
    fixture = _fixture(run.config["variant"], run.config["mode"])
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
    _validate_execution(arrays, fixture.config["mode"])
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
        values, contacts = audit_device_slider_substep(
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


def save_articulated_device_run(run, destination, *, source=None):
    validate_articulated_device_run(run)
    save_run(run, destination, source=source)


def load_articulated_device_run(
    directory, *, require_complete=False, expected_scene=None, expected_mode=None
):
    _check_source_record(directory)
    run = load_run(directory)
    validate_articulated_device_run(
        run,
        require_complete=require_complete,
        expected_scene=expected_scene,
        expected_mode=expected_mode,
    )
    return run
