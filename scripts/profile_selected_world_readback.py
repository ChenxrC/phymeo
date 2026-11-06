"""Measure unchanged selected-world observations; no readback plan or new kernel.

One process measures one (W, K, kind). Timings are instrumented HOST call-boundary
wall times, not GPU event times. Full public outputs are audited after each window.
"""

import argparse
import json
import math
import platform
import re
import subprocess
import sys
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
H = 1 / 128
COUNTERS = (
    "device_allocations",
    "device_owned_bytes",
    "topology_uploaded_bytes",
    "initial_uploaded_bytes",
    "control_uploaded_bytes",
    "restore_uploaded_bytes",
    "status_downloaded_bytes",
    "summary_downloaded_bytes",
    "snapshot_downloaded_bytes",
    "kinematics_downloaded_bytes",
    "launch_attempts",
    "submitted_kernels",
    "synchronizations",
    "completed_steps",
    "accepted_world_steps",
    "rejected_world_steps",
    "failed_steps",
    "recovery_sync_attempts",
)
TIMED_PARTS = ("submit", "sync", "download", "upload", "host_construct")


@dataclass(frozen=True)
class ProfileConfig:
    worlds: int
    selection: int | str
    kind: str
    mode: str
    warm_windows: int = 5
    formal_windows: int = 20
    steps: int = 32

    def __post_init__(self):
        if type(self.worlds) is not int or self.worlds not in (1, 32, 128):
            raise ValueError("worlds must be 1, 32 or 128")
        if self.selection != "all" and (
            type(self.selection) is not int
            or self.selection not in (1, 8)
            or self.selection > self.worlds
        ):
            raise ValueError("selection must be 1, 8 or all, with K <= W")
        if self.kind not in ("snapshot", "kinematics"):
            raise ValueError("kind must be snapshot or kinematics")
        if self.mode not in ("interpret", "cuda"):
            raise ValueError("backend must be interpret or cuda")
        for name, low, high in (
            ("warm_windows", 0, 5),
            ("formal_windows", 1, 20),
            ("steps", 1, 32),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def matrix_cases():
    """Fourteen independent tasks; W1 K1/all is deliberately represented once."""
    return [
        (worlds, selected, kind)
        for kind in ("snapshot", "kinematics")
        for worlds in (1, 32, 128)
        for selected in (("all",) if worlds == 1 else (1, 8, "all"))
    ]


def _git(root, *arguments):
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def freeze_files(paths):
    return {
        str(Path(path).resolve()): sha256(Path(path).read_bytes()).hexdigest() for path in paths
    }


def verify_files(frozen):
    for name, expected in frozen.items():
        path = Path(name)
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"source file changed or disappeared: {name}")


def bind_source(root, expected_head):
    root = Path(root).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", expected_head):
        raise ValueError("expected HEAD must be a full lowercase Git SHA")
    actual = _git(root, "rev-parse", "HEAD")
    if actual != expected_head:
        raise RuntimeError(f"source HEAD mismatch: expected {expected_head}, found {actual}")
    targets = [
        "src/phymeo",
        "scripts/profile_selected_world_readback.py",
        "tests/scale/test_selected_world_readback.py",
        "pyproject.toml",
    ]
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *targets)
    if dirty:
        raise RuntimeError(f"source must be committed and clean: {dirty}")
    names = _git(root, "ls-files", "--", *targets).splitlines()
    if "scripts/profile_selected_world_readback.py" not in names:
        raise RuntimeError("measurement script is not in the bound source commit")
    return dict(
        root=str(root),
        head=actual,
        tracked_paths=names,
        files=freeze_files(root / name for name in names),
    )


def verify_source(binding):
    if _git(binding["root"], "rev-parse", "HEAD") != binding["head"]:
        raise RuntimeError("source HEAD changed during measurement")
    verify_files(binding["files"])
    current = bind_source(binding["root"], binding["head"])
    if current != binding:
        raise RuntimeError("source manifest changed during measurement")


def reserve_output(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("output must be an absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("x", encoding="utf8", newline="\n")


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def frames_sha256(frames):
    return sha256(
        json.dumps(frames, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")
    ).hexdigest()


class PhaseProbe:
    """Temporary wrappers; neither add synchronization nor change returned values.

    Nested methods here are disjoint (launch/sync/read/upload/constructor). The
    remainder includes HOST validation, row selection, dictionary construction and
    the wrappers' own bookkeeping, so it is never labelled pure constructor time.
    """

    def __init__(self, batch):
        self.batch = batch
        self.phase = None
        self.snapshot_reads = []
        self.data = {name: Counter() for name in ("step", "observation")}
        self.hooks = ExitStack()

    def _wrap(self, original, part):
        def measured(*args, **kwargs):
            if self.phase is None:
                return original(*args, **kwargs)
            row = self.data[self.phase]
            row[part + "_attempts"] += 1
            begin = perf_counter_ns()
            try:
                result = original(*args, **kwargs)
            finally:
                row[part + "_ns"] += perf_counter_ns() - begin
            row[part + "_calls"] += 1
            if part == "download":
                row["read_bytes"] += result.nbytes
                if self.phase == "observation" and args[1] == "snapshot":
                    self.snapshot_reads.append(result)
                row["d2h_bytes"] += result.nbytes if self.batch._mode == "cuda" else 0
            elif part == "upload":
                row["h2d_bytes"] += args[1].nbytes if self.batch._mode == "cuda" else 0
            return result

        return measured

    def __enter__(self):
        from phymeo.backends import _world_kinematics, _world_step_layout

        try:
            for method, part in (
                ("_launch", "submit"),
                ("_sync", "sync"),
                ("_read", "download"),
                ("_upload", "upload"),
            ):
                self.hooks.enter_context(
                    patch.object(self.batch, method, self._wrap(getattr(self.batch, method), part))
                )
            cls = _world_step_layout.WorldBinding
            self.hooks.enter_context(
                patch.object(cls, "checkpoint", self._wrap(cls.checkpoint, "host_construct"))
            )
            self.hooks.enter_context(
                patch.object(
                    _world_kinematics,
                    "WorldKinematics",
                    self._wrap(_world_kinematics.WorldKinematics, "host_construct"),
                )
            )
        except BaseException:
            self.hooks.close()
            raise
        return self

    def __exit__(self, *exc):
        return self.hooks.__exit__(*exc)

    def call(self, phase, function, *args):
        if phase not in self.data or self.phase is not None:
            raise ValueError("invalid or nested profile phase")
        self.phase = phase
        begin = perf_counter_ns()
        try:
            return function(*args)
        finally:
            self.data[phase]["wall_ns"] += perf_counter_ns() - begin
            self.phase = None

    def phases(self):
        result = {}
        for phase, row in self.data.items():
            item = {
                name: int(row[name])
                for name in (
                    "wall_ns",
                    "read_bytes",
                    "d2h_bytes",
                    "h2d_bytes",
                    *(
                        part + suffix
                        for part in TIMED_PARTS
                        for suffix in ("_ns", "_calls", "_attempts")
                    ),
                )
            }
            item["host_other_ns"] = item["wall_ns"] - sum(
                item[part + "_ns"] for part in TIMED_PARTS
            )
            if item["host_other_ns"] < 0:
                raise RuntimeError("profile phase accounting overlapped")
            result[phase] = item
        return result


def make_worlds(count):
    """Actual 2 kg spheres and delayed 2 kg one-DOF driven robot worlds."""
    import numpy as np

    import phymeo as ph
    from phymeo.backends.tyx_world_batch import FreeSphere, PrismaticSlider
    from phymeo.solvers.driven_robot import DrivenRobotProgram

    model = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    program = DrivenRobotProgram(model, kp=[0.0], kd=[0.0], effort_limit=[4.0], delay_steps=1)
    initial = program.initial_state(ph.RobotState(np.zeros(1), np.zeros(1)))
    command = program.inputs(np.zeros(1), feedforward=np.array([2.0]))
    worlds, controls, description = [], {}, []
    for i in range(count):
        key = 101 + 17 * i
        if i % 2:
            worlds.append((key, PrismaticSlider(program, initial)))
            controls[key] = command
            description.append(
                dict(
                    key=key,
                    slot=i,
                    law="delayed_prismatic",
                    mass_kg=2,
                    axis=[1, 0, 0],
                    q_m=0,
                    qd_m_s=0,
                    kp=0,
                    kd=0,
                    gravity=[0, 0, 0],
                    delay_steps=1,
                    effort_limit_n=4,
                    held_control=[0, 0, 2, 0],
                    initial_history=[0, 0, 0],
                )
            )
        else:
            position = (i / 8, -i / 16, 2.0)
            velocity = (0.25, -0.125, 0.0)
            worlds.append(
                (
                    key,
                    FreeSphere(
                        2.0, 0.125, position=position, velocity=velocity, gravity=(0, 0, -8)
                    ),
                )
            )
            description.append(
                dict(
                    key=key,
                    slot=i,
                    law="free_sphere",
                    mass_kg=2,
                    radius_m=0.125,
                    position_m=position,
                    velocity_m_s=velocity,
                    gravity_m_s2=[0, 0, -8],
                    held_force_n=[0, 0, 0],
                )
            )
    return worlds, controls, _plain(description)


def _canonical_output(output, kind):
    if kind == "snapshot":
        return dict(checkpoints=[_plain(asdict(point)) for point in output.values()])
    if output.values.flags.writeable or output.ticks.flags.writeable:
        raise AssertionError("kinematics output is not read-only")
    if str(output.values.dtype) != "float32" or str(output.ticks.dtype) != "int32":
        raise AssertionError("kinematics output precision changed")
    return dict(
        keys=list(output.keys),
        widths=list(output.widths),
        ticks=output.ticks.tolist(),
        values=output.values.tolist(),
    )


def _audit_output(output, kind, keys, tick):
    """Closed-form dyadic semi-implicit motion, independent of the integration code."""
    positions, velocities = [], []
    if kind == "snapshot":
        rows = output["checkpoints"]
        if [row["world"] for row in rows] != list(keys):
            raise AssertionError("checkpoint order or world ownership changed")
    else:
        if output["keys"] != list(keys) or output["ticks"] != [tick] * len(keys):
            raise AssertionError("kinematics order or accepted tick changed")
    for index, key in enumerate(keys):
        slot = (key - 101) // 17
        if slot % 2:
            position = [tick * (tick - 1) / 32768]
            velocity = [(tick - 1) / 128]
            effort = [0.0 if tick == 1 else 2.0]
            history, control = [0.0, 0.0, 2.0], [0.0, 0.0, 2.0, 0.0]
            topology = [2, "robot", [1.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 2.0]]
        else:
            position = [
                slot / 8 + tick / 512,
                -slot / 16 - tick / 1024,
                2 - tick * (tick + 1) / 4096,
            ]
            velocity = [0.25, -0.125, -tick / 16]
            effort, history, control = [0.0] * 3, [], [0.0] * 3
            topology = [0, "", [], [0.5, 0.0, 0.0, -8.0, 0.0, 0.0, 0.0, 0.125, 2.0]]
        width = len(position)
        if kind == "snapshot":
            row = rows[index]
            if (
                row["tick"] != tick
                or row["applied_effort"] != effort
                or row["history"] != history
                or row["control"] != control
                or row["topology"] != topology
            ):
                raise AssertionError(f"full checkpoint/history mismatch at {key}, tick {tick}")
            actual_p, actual_v = row["position"], row["velocity"]
        else:
            if output["widths"][index] != width:
                raise AssertionError("kinematics dimension changed")
            values = output["values"][index]
            if len(values) != 6 or any(values[width:3]) or any(values[3 + width :]):
                raise AssertionError("kinematics padding or width changed")
            actual_p, actual_v = values[:width], values[3 : 3 + width]
        if len(actual_p) != width or len(actual_v) != width:
            raise AssertionError("physical state width changed")
        positions.extend(abs(a - b) for a, b in zip(actual_p, position, strict=True))
        velocities.extend(abs(a - b) for a, b in zip(actual_v, velocity, strict=True))
    ep, ev = max(positions, default=0.0), max(velocities, default=0.0)
    if not math.isfinite(ep + ev) or ep != 0 or ev != 0:
        raise AssertionError(f"dyadic physical oracle mismatch: {ep} m, {ev} m/s")
    return ep, ev


def _counters(batch):
    import tyx

    current = batch.stats()
    return ({name: int(current[name]) for name in COUNTERS}, asdict(tyx.runtime_stats()))


def _delta(after, before):
    return {name: after[name] - value for name, value in before.items()}


def _audit_counters(config, count, steps, phases, counters, native, cold):
    """Check recorded transfers, not inferred/fabricated measurements."""
    cuda = config.mode == "cuda"
    reads = 52 * steps * (config.worlds if config.kind == "kinematics" else count)
    observed_launches = steps * count if config.kind == "snapshot" and count < config.worlds else 0
    if (
        phases["observation"]["read_bytes"] != reads
        or phases["observation"]["submit_calls"] != observed_launches
        or phases["observation"]["sync_calls"] != observed_launches
    ):
        raise AssertionError("observed readback calls/bytes do not match the existing API")
    if counters["accepted_world_steps"] != config.worlds * steps:
        raise AssertionError("not every physical world step was accepted")
    for name in (
        "rejected_world_steps",
        "failed_steps",
        "recovery_sync_attempts",
        "device_allocations",
        "topology_uploaded_bytes",
        "restore_uploaded_bytes",
    ):
        if counters[name] != 0:
            raise AssertionError(f"unexpected steady workspace activity: {name}")
    if not cold and counters["control_uploaded_bytes"] != 0:
        raise AssertionError("held input was uploaded again in a warm window")
    if counters["submitted_kernels"] != steps + observed_launches:
        raise AssertionError("unexpected application kernel count")
    if counters["synchronizations"] != steps + observed_launches:
        raise AssertionError("unexpected explicit synchronization count")
    actual_reads = sum(p["d2h_bytes"] for p in phases.values())
    actual_writes = sum(p["h2d_bytes"] for p in phases.values())
    if counters[config.kind + "_downloaded_bytes"] != (reads if cuda else 0):
        raise AssertionError("public readback byte counter disagrees with actual reads")
    if counters["status_downloaded_bytes"] != (4 * config.worlds * steps if cuda else 0):
        raise AssertionError("acceptance readback was hidden or changed")
    if (
        actual_reads
        != counters["status_downloaded_bytes"] + counters[config.kind + "_downloaded_bytes"]
    ):
        raise AssertionError("download attribution does not balance")
    if actual_writes != counters["control_uploaded_bytes"]:
        raise AssertionError("upload attribution does not balance")
    expected_native = dict(
        cuda_d2h_bytes=actual_reads,
        cuda_h2d_bytes=actual_writes,
        cuda_kernel_launches=counters["submitted_kernels"] if cuda else 0,
        cuda_d2d_bytes=0,
    )
    extra = {name: native[name] - value for name, value in expected_native.items()}
    # Cold runtime initialization is visible separately; no unexplained warm activity.
    if any(value < 0 for value in extra.values()) or (not cold and any(extra.values())):
        raise AssertionError(f"native and batch counters disagree: {extra}")
    return extra


def _window(batch, config, keys, start_tick, steps, *, cold=False):
    retained = []
    selected = None if config.selection == "all" else keys
    before, native_before = _counters(batch)
    with PhaseProbe(batch) as probe:
        begin = perf_counter_ns()
        for offset in range(steps):
            accepted = probe.call("step", batch.step, H)
            first_read = len(probe.snapshot_reads)
            output = probe.call("observation", getattr(batch, config.kind), selected)
            wire = tuple(probe.snapshot_reads[first_read:])
            retained.append((start_tick + offset + 1, accepted, output, wire))
        whole_ns = perf_counter_ns() - begin
    after, native_after = _counters(batch)
    phases = probe.phases()
    exclusive = {
        phase + "_" + part: data[part + "_ns"]
        for phase, data in phases.items()
        for part in (*TIMED_PARTS, "host_other")
    }
    exclusive["loop_store_and_dispatch"] = whole_ns - sum(p["wall_ns"] for p in phases.values())
    if min(exclusive.values()) < 0 or sum(exclusive.values()) != whole_ns:
        raise RuntimeError("exclusive HOST wall accounting failed")
    audit_begin = perf_counter_ns()
    frames, errors = [], []
    all_keys = tuple(batch._slots)
    for tick, accepted, output, wire in retained:
        if tuple(accepted) != all_keys or not all(accepted.values()):
            raise AssertionError("physical world rejection or ownership mismatch")
        canonical = _canonical_output(output, config.kind)
        errors.append(_audit_output(canonical, config.kind, keys, tick))
        frame = dict(tick=tick, accepted=[[k, v] for k, v in accepted.items()], output=canonical)
        if config.kind == "snapshot":
            frame["wire_snapshot"] = [
                dict(dtype=str(row.dtype), shape=list(row.shape), values=row.tolist())
                for row in wire
            ]
        frames.append(frame)
    counters, native = _delta(after, before), _delta(native_after, native_before)
    extra = _audit_counters(config, len(keys), steps, phases, counters, native, cold)
    return dict(
        start_tick=start_tick,
        end_tick=start_tick + steps,
        steps=steps,
        whole_ns=whole_ns,
        phases=phases,
        exclusive_ns=exclusive,
        batch_counter_delta=counters,
        native_counter_delta=native,
        cold_native_extra=extra,
        counter_audit="passed",
        frames=frames,
        output_sha256=frames_sha256(frames),
        max_position_error_m=max(e[0] for e in errors),
        max_velocity_error_m_s=max(e[1] for e in errors),
        offline_audit_ns=perf_counter_ns() - audit_begin,
    )


def _quantiles(values):
    return dict(
        p50_ms=median(values) / 1e6, p95_ms=sorted(values)[math.ceil(0.95 * len(values)) - 1] / 1e6
    )


def _summary(config, selected_count, windows):
    total = sum(w["whole_ns"] for w in windows)
    observations = sum(w["phases"]["observation"]["wall_ns"] for w in windows)
    download = sum(w["phases"]["observation"]["download_ns"] for w in windows)
    submit_sync = sum(
        w["phases"]["observation"][p + "_ns"] for w in windows for p in ("submit", "sync")
    )
    steps = sum(w["steps"] for w in windows)
    exclusive = {
        name: sum(w["exclusive_ns"][name] for w in windows) for name in windows[0]["exclusive_ns"]
    }
    result = dict(
        formal_windows=len(windows),
        physical_steps=steps,
        whole_window=_quantiles([w["whole_ns"] for w in windows]),
        observation_window=_quantiles([w["phases"]["observation"]["wall_ns"] for w in windows]),
        all_world_substeps=steps * config.worlds,
        selected_world_observations=steps * selected_count,
        world_substeps_per_s=steps * config.worlds * 1e9 / total,
        selected_world_observations_per_s=steps * selected_count * 1e9 / total,
        batch_counter_totals={
            name: sum(w["batch_counter_delta"][name] for w in windows) for name in COUNTERS
        },
        native_counter_totals={
            name: sum(w["native_counter_delta"][name] for w in windows)
            for name in windows[0]["native_counter_delta"]
        },
        exclusive_total_ns=exclusive,
        exclusive_fraction_of_whole={name: ns / total for name, ns in exclusive.items()},
        observation_download_fraction=download / observations,
        observation_submit_sync_fraction=submit_sync / observations,
        observation_readback_fraction=(download + submit_sync) / observations,
        observation_download_fraction_of_whole=download / total,
        observation_submit_sync_fraction_of_whole=submit_sync / total,
    )
    eligible = (
        config.mode == "cuda"
        and selected_count < config.worlds
        and config.warm_windows == 5
        and config.formal_windows == 20
        and config.steps == 32
        and len(windows) == 20
        and steps == 640
    )
    gate = dict(
        eligible_formal_cuda_subset=eligible,
        threshold=0.15,
        denominator="sum of formal whole_ns: advance plus same-kind observation, including HOST construction",
        download_meets_threshold=download / total >= 0.15,
        submit_and_sync_meets_threshold=submit_sync / total >= 0.15,
        measured_cost_gate=eligible and max(download, submit_sync) / total >= 0.15,
        action="parent review required; this script never creates or enables a new kernel",
    )
    return result, gate


def _device_signature(batch):
    if batch._mode == "interpret":
        return dict(backend="interpret", actual_device="HOST", real_cuda=False)
    from cuda.bindings import driver as cu

    def checked(result):
        if result[0] != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA identity query failed: {result[0]}")
        return result[1]

    ordinal = int(batch._states[0].device_ordinal)
    if any(int(b.device_ordinal) != ordinal for b in (*batch._states, *batch._ticks)):
        raise RuntimeError("accepted-state buffers span different CUDA devices")
    device = checked(cu.cuDeviceGet(ordinal))
    name = checked(cu.cuDeviceGetName(128, device)).decode("utf8").rstrip("\x00")
    return dict(
        backend="cuda",
        actual_device=f"cuda:{ordinal}",
        real_cuda=True,
        device_name=name,
        driver_version=int(checked(cu.cuDriverGetVersion())),
    )


def run_profile(config, *, cache_root, on_window=None, required_source_root=None):
    """Run one physical trajectory. Callers control serial execution externally."""
    imports_begin = perf_counter_ns()
    import tyx

    import phymeo
    from phymeo.backends.tyx_world_batch import TyxWorldBatch

    import_ns = perf_counter_ns() - imports_begin
    if required_source_root is not None:
        expected = Path(required_source_root) / "src" / "phymeo" / "__init__.py"
        if Path(phymeo.__file__).resolve() != expected.resolve():
            raise RuntimeError(f"wrong imported source: {phymeo.__file__}")
    case_begin = perf_counter_ns()
    worlds, controls, inputs = make_worlds(config.worlds)
    case_ns = perf_counter_ns() - case_begin
    keys = tuple(k for k, _ in worlds)
    selected = keys if config.selection == "all" else tuple(reversed(keys))[: config.selection]
    construction_begin = perf_counter_ns()
    batch = TyxWorldBatch(worlds, config.mode, cache_root=cache_root)
    construction_ns = perf_counter_ns() - construction_begin
    try:
        initial_counts, initial_native = _counters(batch)
        controls_begin = perf_counter_ns()
        batch.set_controls(controls)
        control_setup_ns = perf_counter_ns() - controls_begin
        collected = {"warmups": [], "measurements": []}
        tick = 0
        for index in range(1 + config.warm_windows + config.formal_windows):
            label = (
                "cold"
                if index == 0
                else ("warmups" if index <= config.warm_windows else "measurements")
            )
            steps = 1 if index == 0 else config.steps
            window = _window(batch, config, selected, tick, steps, cold=index == 0)
            tick += steps
            if label == "cold":
                collected[label] = window
            else:
                collected[label].append(window)
            if on_window is not None:
                on_window(label, index, window)
        summary, gate = _summary(config, len(selected), collected["measurements"])
        signature = _device_signature(batch)
        if (
            config.mode == "cuda"
            and sum(
                w["native_counter_delta"]["cuda_kernel_launches"] for w in collected["measurements"]
            )
            <= 0
        ):
            raise RuntimeError("CUDA was requested but no real device launch was recorded")
        return dict(
            schema="p65-selected-world-readback-v1",
            status="complete",
            config=asdict(config),
            dt_s=H,
            selected_keys=list(selected),
            inputs=inputs,
            final_tick=tick,
            state_policy="one continuous trajectory: cold 1 step, then warm and formal windows; no reset",
            scope="instrumented HOST API wall time; asynchronous submit is not GPU kernel time",
            output_contract="complete public checkpoints"
            if config.kind == "snapshot"
            else "world keys, dimensions, exact int32 ticks and six FP32 kinematic components",
            timing_excludes="case/import/setup, per-window raw serialization and independent analytic audit",
            setup=dict(
                import_ns=import_ns,
                case_ns=case_ns,
                batch_construct_ns=construction_ns,
                control_setup_ns=control_setup_ns,
                initial_batch_counts=initial_counts,
                initial_native_counts=initial_native,
            ),
            runtime=dict(
                python=sys.version,
                platform=platform.platform(),
                phymeo=version("phymeo"),
                tyx=version("tyx-compute"),
                phymeo_file=str(Path(phymeo.__file__).resolve()),
                tyx_file=str(Path(tyx.__file__).resolve()),
                **signature,
            ),
            audit=dict(
                max_position_error_m=0.0,
                max_velocity_error_m_s=0.0,
                reference="closed-form semi-implicit sphere and one-step-delayed constant slider force",
                full_checkpoint_history_checked=config.kind == "snapshot",
            ),
            summary=summary,
            gate=gate,
            **collected,
        )
    finally:
        batch.close()


PLAN_COUNTERS = (
    "readback_index_uploaded_bytes", "readback_plan_owned_bytes",
    "readback_plan_reads", "readback_plan_failed_reads",
)


def _plan_counters(batch):
    current = batch.stats()
    return {name: int(current[name]) for name in PLAN_COUNTERS}


def _public_frames(frames):
    # C0's raw wire shape is deliberately not a cross-strategy output identity.
    return [{name: frame[name] for name in ("tick", "accepted", "output")} for frame in frames]


def _comparison_window(batch, config, keys, start_tick, steps, *, plan=None, cold=False):
    """No call-boundary wrappers in the actual A/B application timing loop."""
    before, native_before = _counters(batch)
    plan_before = _plan_counters(batch)
    stats_before = None if plan is None else plan.stats()
    read = plan.read if plan is not None else getattr(batch, config.kind)
    selected = None if config.selection == "all" else keys
    retained = []
    begin = perf_counter_ns()
    for offset in range(steps):
        accepted = batch.step(H)
        output = read() if plan is not None else read(selected)
        retained.append((start_tick + offset + 1, accepted, output))
    whole_ns = perf_counter_ns() - begin
    after, native_after = _counters(batch)
    plan_after = _plan_counters(batch)
    stats_after = None if plan is None else plan.stats()
    audit_begin = perf_counter_ns()
    frames = []
    for tick, accepted, output in retained:
        if tuple(accepted) != tuple(batch._slots) or not all(accepted.values()):
            raise AssertionError("comparison world rejection or ownership mismatch")
        canonical = _canonical_output(output, config.kind)
        _audit_output(canonical, config.kind, keys, tick)
        frames.append(dict(tick=tick, accepted=[[k, v] for k, v in accepted.items()],
                           output=canonical))
    counts, native = _delta(after, before), _delta(native_after, native_before)
    plan_counts = _delta(plan_after, plan_before)
    cuda = config.mode == "cuda"
    k = len(keys)
    if plan is None:
        read_bytes = 52 * (config.worlds if config.kind == "kinematics" else k) * steps
        read_launches = steps * k if config.kind == "snapshot" and k < config.worlds else 0
        transfers = steps * (1 + (2 * k if read_launches else 2))
    else:
        read_bytes = (28 if config.kind == "kinematics" else 52) * k * steps
        read_launches = steps
        transfers = 3 * steps
    expected = {
        "completed_steps": steps, "accepted_world_steps": steps * config.worlds,
        "submitted_kernels": steps + read_launches,
        "synchronizations": steps + read_launches,
        "status_downloaded_bytes": 4 * config.worlds * steps if cuda else 0,
        config.kind + "_downloaded_bytes": read_bytes if cuda else 0,
    }
    if any(counts[name] != value for name, value in expected.items()):
        raise AssertionError(f"comparison application counts changed: {counts}")
    zero_fields = ("rejected_world_steps", "failed_steps", "recovery_sync_attempts",
                   "device_allocations", "topology_uploaded_bytes", "restore_uploaded_bytes")
    if any(counts[name] for name in zero_fields):
        raise AssertionError("unexpected comparison steady allocation, rejection or upload")
    if not cold and counts["control_uploaded_bytes"]:
        raise AssertionError("warm comparison re-uploaded held controls")
    if plan_counts != dict(
        readback_index_uploaded_bytes=0, readback_plan_owned_bytes=0,
        readback_plan_reads=steps if plan is not None else 0, readback_plan_failed_reads=0,
    ):
        raise AssertionError(f"comparison plan accounting changed: {plan_counts}")
    if plan is not None:
        if stats_after["downloaded_bytes"] - stats_before["downloaded_bytes"] != (read_bytes if cuda else 0):
            raise AssertionError("plan read bytes disagree with the public counter")
        if stats_after["device_allocations"] != stats_before["device_allocations"]:
            raise AssertionError("plan allocated during the measured window")
    expected_native = dict(
        cuda_kernel_launches=(steps + read_launches) if cuda else 0,
        cuda_d2h_bytes=(4 * config.worlds * steps + read_bytes) if cuda else 0,
        cuda_h2d_bytes=counts["control_uploaded_bytes"], cuda_d2d_bytes=0,
        cuda_d2h_transfers=transfers if cuda else 0,
    )
    extra = {name: native[name] - value for name, value in expected_native.items()}
    if any(value < 0 for value in extra.values()) or (not cold and any(extra.values())):
        raise AssertionError(f"comparison native counter discrepancy: {extra}")
    return dict(
        start_tick=start_tick, end_tick=start_tick + steps, steps=steps, whole_ns=whole_ns,
        timing_mode="unwrapped advance plus same-kind public observation; no PhaseProbe",
        batch_counter_delta=counts, native_counter_delta=native,
        plan_counter_delta=plan_counts, plan_stats_before=stats_before,
        plan_stats_after=stats_after, cold_native_extra=extra,
        counter_audit="passed", frames=frames, public_output_sha256=frames_sha256(frames),
        max_position_error_m=0.0, max_velocity_error_m_s=0.0,
        offline_audit_ns=perf_counter_ns() - audit_begin,
    )


def _comparison_summary(config, selected_count, windows):
    lanes = {}
    for lane in ("original", "plan"):
        rows = [window["lanes"][lane] for window in windows]
        steps = sum(row["steps"] for row in rows)
        total = sum(row["whole_ns"] for row in rows)
        lanes[lane] = dict(
            **_quantiles([row["whole_ns"] for row in rows]),
            timed_total_ns=total, all_world_substeps=steps * config.worlds,
            selected_world_observations=steps * selected_count,
            world_substeps_per_s=steps * config.worlds * 1e9 / total,
            selected_world_observations_per_s=steps * selected_count * 1e9 / total,
            batch_counter_totals={name: sum(row["batch_counter_delta"][name] for row in rows)
                                  for name in COUNTERS},
            native_counter_totals={name: sum(row["native_counter_delta"][name] for row in rows)
                                   for name in rows[0]["native_counter_delta"]},
            plan_counter_totals={name: sum(row["plan_counter_delta"][name] for row in rows)
                                 for name in PLAN_COUNTERS},
        )
    ratios = [window["lanes"]["original"]["whole_ns"] / window["lanes"]["plan"]["whole_ns"]
              for window in windows]
    return dict(
        lanes=lanes, per_window_original_over_plan=ratios,
        median_paired_original_over_plan=median(ratios),
        p50_original_over_plan=lanes["original"]["p50_ms"] / lanes["plan"]["p50_ms"],
        p95_original_over_plan=lanes["original"]["p95_ms"] / lanes["plan"]["p95_ms"],
        throughput_plan_over_original=lanes["plan"]["world_substeps_per_s"]
                                      / lanes["original"]["world_substeps_per_s"],
        ratio_scope="same public kind, W/K, physical trajectory, accepted ticks and complete outputs",
        default_changed=False,
    )


def run_comparison(config, *, cache_root, first="original", on_window=None,
                   required_source_root=None):
    if first not in ("original", "plan"):
        raise ValueError("first strategy must be original or plan")
    import phymeo
    import tyx
    from phymeo.backends.tyx_world_batch import TyxWorldBatch

    if required_source_root is not None:
        expected = Path(required_source_root) / "src" / "phymeo" / "__init__.py"
        if Path(phymeo.__file__).resolve() != expected.resolve():
            raise RuntimeError(f"wrong imported source: {phymeo.__file__}")
    worlds, controls, inputs = make_worlds(config.worlds)
    keys = tuple(key for key, _ in worlds)
    selected = keys if config.selection == "all" else tuple(reversed(keys))[:config.selection]
    first_order = (first, "plan" if first == "original" else "original")
    setups, batches, plans = {}, {}, {}
    with ExitStack() as resources:
        for lane in first_order:
            started = perf_counter_ns()
            batch = resources.enter_context(TyxWorldBatch(
                worlds, config.mode, cache_root=Path(cache_root) / lane,
            ))
            constructed_ns = perf_counter_ns() - started
            batch.set_controls(controls)
            prepare_begin = perf_counter_ns()
            plan = batch.prepare_readback(selected, kind=config.kind) if lane == "plan" else None
            prepare_ns = perf_counter_ns() - prepare_begin
            batches[lane], plans[lane] = batch, plan
            setups[lane] = dict(batch_construct_ns=constructed_ns, prepare_plan_ns=prepare_ns,
                                batch_stats=batch.stats(), plan_stats=None if plan is None else plan.stats())
        collected = dict(warmups=[], measurements=[])
        tick = 0
        for index in range(1 + config.warm_windows + config.formal_windows):
            label = "cold" if index == 0 else (
                "warmups" if index <= config.warm_windows else "measurements"
            )
            order = first_order if index % 2 == 0 else tuple(reversed(first_order))
            steps = 1 if index == 0 else config.steps
            pair = {}
            for lane in order:
                pair[lane] = _comparison_window(
                    batches[lane], config, selected, tick, steps,
                    plan=plans[lane], cold=index == 0,
                )
            original_frames = _public_frames(pair["original"]["frames"])
            plan_frames = _public_frames(pair["plan"]["frames"])
            if original_frames != plan_frames:
                raise AssertionError("complete public outputs differ between original and plan")
            if pair["original"]["public_output_sha256"] != pair["plan"]["public_output_sha256"]:
                raise AssertionError("same public output has different canonical SHA")
            tick += steps
            window = dict(order=order, end_tick=tick, lanes=pair, same_public_outputs=True,
                          public_output_sha256=pair["original"]["public_output_sha256"])
            if label == "cold":
                collected[label] = window
            else:
                collected[label].append(window)
            if on_window is not None:
                on_window(label, index, window)
        signatures = {lane: _device_signature(batch) for lane, batch in batches.items()}
        if signatures["original"] != signatures["plan"]:
            raise AssertionError("comparison lanes used different device identities")
        summary = _comparison_summary(config, len(selected), collected["measurements"])
        result = dict(
            schema="p65-selected-world-readback-comparison-v1", status="complete",
            config=asdict(config), first=first, dt_s=H, selected_keys=list(selected), inputs=inputs,
            final_tick=tick, same_public_outputs=True, setup=setups, summary=summary,
            state_policy="two independent same-input trajectories; each cold1 + warm + formal; no resets",
            timing_mode="unwrapped physical advance + same-kind public output; no PhaseProbe",
            timing_excludes="construction, plan preparation, counter snapshots, JSON/file IO and offline audit",
            raw_scope="complete public frames; C0 wire receipts remain archived separately",
            runtime=dict(python=sys.version, phymeo=version("phymeo"), tyx=version("tyx-compute"),
                         phymeo_file=str(Path(phymeo.__file__).resolve()),
                         tyx_file=str(Path(tyx.__file__).resolve()), lanes=signatures),
            audit=dict(max_position_error_m=0.0, max_velocity_error_m_s=0.0,
                       full_checkpoint_history_checked=config.kind == "snapshot"),
            **collected,
        )
    result["closed_batch_stats"] = {lane: batch.stats() for lane, batch in batches.items()}
    result["closed_plan_stats"] = plans["plan"].stats()
    if any(batch.stats()["readback_plan_owned_bytes"] or batch.stats()["readback_plan_live_count"]
           for batch in batches.values()):
        raise AssertionError("comparison left plan resources alive after batch close")
    return result

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, choices=(1, 32, 128), required=True)
    parser.add_argument(
        "--select", "--selection", dest="selection", choices=("1", "8", "all"), required=True
    )
    parser.add_argument("--kind", choices=("snapshot", "kinematics"), required=True)
    parser.add_argument(
        "--backend", "--mode", dest="mode", choices=("interpret", "cuda"), required=True
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--compare-plan", action="store_true")
    parser.add_argument("--first", choices=("original", "plan"), default="original")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warm-windows", type=int, default=5)
    parser.add_argument("--formal-windows", type=int, default=20)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args(argv)
    args.selection = "all" if args.selection == "all" else int(args.selection)
    args.config = ProfileConfig(
        args.worlds,
        args.selection,
        args.kind,
        args.mode,
        args.warm_windows,
        args.formal_windows,
        args.steps,
    )
    return args


def _compact_window(window):
    if "lanes" in window:
        return {**{name: value for name, value in window.items() if name != "lanes"},
                "lanes": {lane: _compact_window(row) for lane, row in window["lanes"].items()}}
    return {
        **{name: value for name, value in window.items() if name != "frames"},
        "frame_count": len(window["frames"]),
    }


def main(argv=None):
    args = parse_args(argv)
    if not args.out.is_absolute() or not args.cache_root.is_absolute():
        raise ValueError("output and cache-root must be absolute paths")
    source = bind_source(ROOT, args.expected_head)
    raw_path = args.out.with_suffix(".frames.jsonl")
    if args.out == raw_path:
        raise ValueError("receipt path collides with its raw frame sidecar")
    for path in (args.out, raw_path, args.cache_root):
        if path.exists():
            raise FileExistsError(f"fresh output, raw sidecar and cache-root required: {path}")
    args.cache_root.mkdir(parents=True)
    base = dict(
        schema="p65-selected-world-readback-v1",
        status="running",
        config=asdict(args.config),
        compare_plan=args.compare_plan, first=args.first,
        source=source,
        command=sys.argv,
        output_path=str(args.out),
        raw_frames_path=str(raw_path),
        cache_root=str(args.cache_root),
        cache_existed_before=False,
    )
    completed = []
    with reserve_output(args.out) as receipt, reserve_output(raw_path) as raw:

        def persist(label, index, window):
            entry = dict(phase=label, index=index, window=window)
            raw.write(json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            raw.flush()
            completed.append(dict(phase=label, index=index, **_compact_window(window)))
            print(
                f"{label} {index}: W{args.worlds} {args.kind} tick={window['end_tick']}", flush=True
            )

        try:
            runner = run_comparison if args.compare_plan else run_profile
            extra = {"first": args.first} if args.compare_plan else {}
            result = runner(
                args.config, cache_root=args.cache_root, on_window=persist,
                required_source_root=ROOT, **extra,
            )
            verify_source(source)
            raw.flush()
            compact = dict(result)
            compact["cold"] = _compact_window(result["cold"])
            for name in ("warmups", "measurements"):
                compact[name] = [_compact_window(window) for window in result[name]]
            final = dict(
                base,
                **compact,
                source_verified_after=True,
                raw_frames_sha256=sha256(raw_path.read_bytes()).hexdigest(),
            )
        except BaseException as error:
            raw.flush()
            failure = dict(
                base,
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
                completed_windows=completed,
                raw_frames_sha256=sha256(raw_path.read_bytes()).hexdigest(),
            )
            receipt.write(json.dumps(failure, indent=2, allow_nan=False))
            receipt.flush()
            raise
        receipt.write(json.dumps(final, indent=2, allow_nan=False))
        receipt.flush()
    print(
        json.dumps(
            dict(output=str(args.out), summary=final["summary"], gate=final.get("gate")),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()


